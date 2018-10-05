import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

from utils import constant, text_utils, torch_utils
from model.modules import SoftDotAttention, BasicAttention, LSTMAttention, LinearAttention, \
        DeepAttention
from model.beam import Beam
from utils.torch_utils import set_cuda

class Seq2SeqWithCopyModel(nn.Module):
    """
    A seq2seq model with a simple LSTM as encoder, and a combination of an attentional LSTM decoder
    and a copy network for decoding.
    """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.vocab_size = opt['vocab_size']
        self.emb_dim = opt['emb_dim']
        self.hidden_dim = opt['hidden_dim']
        self.nlayers = opt['num_layers'] # encoder layers, decoder layers = 1
        self.emb_dropout = opt.get('emb_dropout', 0.0)
        self.dropout = opt['dropout']
        self.pad_token = constant.PAD_ID
        self.max_dec_len = opt['max_dec_len']
        self.use_cuda = opt['cuda']
        self.top = opt.get('top', 1e10)
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.use_bg = opt.get('background', False)
        
        print("Building Seq2Seq with Copy model ...")
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim
        
        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.drop = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)
        self.encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, self.nlayers, \
                bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0) # avoid warning

        dec_input_dim = self.emb_dim
        dec_input_dim += self.hidden_dim if self.use_bg else 0
        self.decoder = LSTMAttention(dec_input_dim, self.dec_hidden_dim, \
                attn_type=self.opt.get('attn_type', 'soft'), coverage=self.opt.get('cov', False))
        self.dec2vocab = nn.Linear(self.dec_hidden_dim, self.vocab_size)
        self.combiner = nn.Sequential( # p_gen = sigma(wh + wx + wc)
                nn.Linear(self.dec_hidden_dim*2 + self.emb_dim, 2),
                nn.LogSoftmax(dim=1))

        if self.use_bg:
            self.bg_encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, 1, \
                    bidirectional=True, batch_first=True, dropout=0) # when nlayer=1, rnn dropout does not apply
            self.bg_drop = nn.Dropout(self.dropout)
            self.bg_attn_layer = BasicAttention(self.hidden_dim)

        self.SOS_tensor = torch.LongTensor([constant.SOS_ID])
        self.SOS_tensor = self.SOS_tensor.cuda() if self.use_cuda else self.SOS_tensor

        self.init_weights()
    
    def init_weights(self):
        # initialize embeddings
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == (self.vocab_size, self.emb_dim), \
                    "Input embedding matrix must match size: {} x {}".format(self.vocab_size, self.emb_dim)
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            init_range = constant.EMB_INIT_RANGE
            self.embedding.weight.data.uniform_(-init_range, init_range)
        # decide finetuning
        if self.top <= 0:
            print("Do not finetune embedding layer.")
            self.embedding.weight.requires_grad = False
        elif self.top < self.vocab_size:
            print("Finetune top {} embeddings.".format(self.top))
            self.embedding.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.top))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, inputs, encoder):
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(encoder.num_layers*2, batch_size, self.enc_hidden_dim), requires_grad=False)
        c0 = Variable(torch.zeros(encoder.num_layers*2, batch_size, self.enc_hidden_dim), requires_grad=False)
        return set_cuda(h0, self.use_cuda), set_cuda(c0, self.use_cuda)
    
    def dec_zero_state(self, batch_size):
        h0 = Variable(torch.zeros(batch_size, self.dec_hidden_dim), requires_grad=False)
        c0 = Variable(torch.zeros(batch_size, self.dec_hidden_dim), requires_grad=False)
        return set_cuda(h0, self.use_cuda), set_cuda(c0, self.use_cuda)
   
    def encode(self, enc_inputs, lens, encoder, sort=False):
        if sort: # need to sort inputs by decreasing len
            lens, indices = torch.sort(lens, descending=True)
            enc_inputs = enc_inputs[indices]
        lens = list(lens)
        self.h0, self.c0 = self.zero_state(enc_inputs, encoder)
        
        packed_inputs = nn.utils.rnn.pack_padded_sequence(enc_inputs, lens, batch_first=True)
        packed_h_in, (hn, cn) = encoder(packed_inputs, (self.h0, self.c0))
        h_in, _ = nn.utils.rnn.pad_packed_sequence(packed_h_in, batch_first=True)
        hn = torch.cat((hn[-1], hn[-2]), 1)
        cn = torch.cat((cn[-1], cn[-2]), 1)

        if sort: # unsort every output
            _, unsort_idx = torch.sort(indices)
            h_in = h_in[unsort_idx]
            hn = hn[unsort_idx]
            cn = cn[unsort_idx]
        return h_in, (hn, cn)

    def decode(self, dec_inputs, dec_hidden, ctx, ctx_tokens, ctx_mask=None, bg_h=None, inference=False):
        """
        Every input should be batch first.
        For inference mode, the output will be decided to be either copier prob or decoder prob.
        """
        batch_size = dec_inputs.size(0)
        seq_len = dec_inputs.size(1)
        ctx_len = ctx_tokens.size(1)
        copy_indices = set_cuda(torch.LongTensor(batch_size).zero_() + -1, self.use_cuda) # by default do not hard copy
        
        if self.use_bg:
            dec_inputs_with_bg = torch.cat([dec_inputs, bg_h.unsqueeze(1).expand(-1, dec_inputs.size(1), -1)], dim=2)
        else:
            dec_inputs_with_bg = dec_inputs # no bg
        # attentional decoder
        h, h_tilde, h_c, attn, cov, dec_hidden = self.decoder(dec_inputs_with_bg, dec_hidden, ctx, ctx_mask)
        # vocab prediction layer
        h_tilde_flat = h_tilde.contiguous().view(-1, h_tilde.size(2)) # B*L x dim
        decoder_logits = self.dec2vocab(h_tilde_flat).view(h_tilde.size(0), h_tilde.size(1), -1)
        # force PAD and UNK logits to -inf
        decoder_logits[:,:,constant.PAD_ID] = -constant.INFINITY_NUMBER
        decoder_logits[:,:,constant.UNK_ID] = -constant.INFINITY_NUMBER
        decoder_probs = self.get_prob(decoder_logits) # [B, QT, V]
        decoder_probs = torch.log(decoder_probs + 1e-12)
        
        # copy network
        copier_probs = attn
        copier_probs = torch.log(copier_probs + 1e-12)
        # [B, QT, CT]

        # combine
        h_flat = h.contiguous().view(-1, h.size(2))
        h_c_flat = h_c.contiguous().view(-1, h_c.size(2))
        dec_inputs_flat = dec_inputs.contiguous().view(-1, dec_inputs.size(2))
        c = self.combiner(torch.cat([h_flat, h_c_flat, dec_inputs_flat], dim=1)).view(batch_size, seq_len, -1) # [B, QT, 2]
        cpy_prob = c[:,:,0]
        dec_prob = c[:,:,1]

        # if inference=True, do hard selection
        if inference:
            assert cpy_prob.size(1) == seq_len == 1, "Inference mode has to decode one step at a time."
            use_cpy = (cpy_prob > dec_prob).float()
            cpy_prob = use_cpy
            dec_prob = 1.0 - use_cpy
            # find copy indices
            _, copy_indices = copier_probs.squeeze(1).max(dim=1)
            for i,c in enumerate(use_cpy.squeeze(1).data):
                if c == 0:
                    copy_indices[i] = -1 # do not hard copy
            copy_indices = copy_indices.data

        expanded_cpy_prob = cpy_prob.unsqueeze(2).expand(batch_size, seq_len, ctx_len)
        expanded_dec_prob = dec_prob.unsqueeze(2).expand(batch_size, seq_len, self.vocab_size)
        
        # scatter probs back
        full_copier_probs = set_cuda(Variable(torch.zeros(batch_size, seq_len, self.vocab_size) + -1e10), \
                self.use_cuda)
        expanded_ctx_tokens = ctx_tokens.unsqueeze(1).expand_as(copier_probs)
        combined_copier_probs = copier_probs + expanded_cpy_prob # combine before scatter
        full_copier_probs.scatter_(2, expanded_ctx_tokens, combined_copier_probs) # scatter info back

        # combine in log space
        combined_probs = torch.exp(full_copier_probs) + torch.exp(expanded_dec_prob + decoder_probs)
        log_probs = torch.log(combined_probs)
        
        # output dec_hidden for future decoding
        if inference:
            return log_probs, dec_hidden, copy_indices
        return log_probs, dec_hidden, attn, cov
       
    def forward(self, src, tgt_in, bg):
        """
        Calculate log probabilities for outputs, combining an attentional decoder and a copy network.
        """
        enc_inputs = self.emb_drop(self.embedding(src))
        dec_inputs = self.emb_drop(self.embedding(tgt_in))
        src_mask = src.data.eq(constant.PAD_ID)
        src_lens = src_mask.eq(0).long().sum(1)
        
        # first encoding
        h_in, enc_hidden = self.encode(enc_inputs, src_lens, encoder=self.encoder)
        dec_hidden = enc_hidden

        if self.use_bg:
            bg_inputs = self.emb_drop(self.embedding(bg))
            bg_mask = bg.data.eq(constant.PAD_ID)
            bg_lens = bg_mask.eq(0).long().sum(1)
            bg_out, bg_hidden = self.encode(bg_inputs, bg_lens, encoder=self.bg_encoder, sort=True)
            bg_h = bg_hidden[0]
            # use attentional representation
            _, bg_h, _ = self.bg_attn_layer(enc_hidden[0], bg_out, mask=bg_mask)
            bg_h = self.bg_drop(bg_h)
        else:
            bg_h = None

        # then decoding
        out_log_probs, _, attn, cov = self.decode(dec_inputs, dec_hidden, h_in, src, ctx_mask=src_mask, bg_h=bg_h)
        return out_log_probs, attn, cov

    def get_prob(self, logits):
        """ Get the probs from logits. """
        logits_reshape = logits.view(-1, self.vocab_size)
        probs = F.softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return probs
        return probs.view(logits.size(0), logits.size(1), -1)

    def get_log_prob(self, logits):
        """ Get the log probs from logits. """
        logits_reshape = logits.view(-1, self.vocab_size)
        probs = F.log_softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return probs
        return probs.view(logits.size(0), logits.size(1), -1)

    def predict(self, src, bg, beam_size=5):
        """ Predict with beam search. """
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        src_mask = src.data.eq(constant.PAD_ID)
        src_lens = list(src_mask.eq(0).long().sum(1))
       
        # (1) encode context
        h_in, enc_hidden = self.encode(enc_inputs, src_lens, encoder=self.encoder)
        (hn, cn) = enc_hidden

        if self.use_bg:
            bg_inputs = self.emb_drop(self.embedding(bg))
            bg_mask = bg.data.eq(constant.PAD_ID)
            bg_lens = bg_mask.eq(0).long().sum(1)
            bg_out, bg_hidden = self.encode(bg_inputs, bg_lens, encoder=self.bg_encoder, sort=True)
            bg_h = bg_hidden[0]
            # use attentional representation
            _, bg_h, _ = self.bg_attn_layer(enc_hidden[0], bg_out, mask=bg_mask)
            bg_h = self.bg_drop(bg_h)
        else:
            bg_h = None

        # (2) set up beam
        with torch.no_grad():
            src_mask = src_mask.repeat(beam_size, 1)
            src = src.repeat(beam_size, 1)
            h_in = h_in.data.repeat(beam_size, 1, 1)
            hn = hn.data.repeat(beam_size, 1)
            cn = cn.data.repeat(beam_size, 1)
            if self.use_bg:
                bg_h = bg_h.data.repeat(beam_size, 1)
        dec_hidden = (hn, cn)
        beam = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]

        def update_state(states, idx, positions, beam_size):
            for e in states:
                br, d = e.size()
                s = e.contiguous().view(beam_size, br // beam_size, d)[:,idx] # locate the batch
                s.data.copy_(s.data.index_select(0, positions))
        
        # (3) main loop
        for i in range(self.max_dec_len):
            dec_inputs = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            dec_inputs = self.embedding(dec_inputs)
            log_probs, dec_hidden, copy_indices = self.decode(dec_inputs, dec_hidden, h_in, src, ctx_mask=src_mask, bg_h=bg_h, inference=True)
            log_probs = log_probs.view(beam_size, batch_size, -1).transpose(0,1).contiguous()
            copy_indices = copy_indices.view(beam_size, batch_size).transpose(0,1).contiguous()

            # advance each beam
            done = []
            for b in range(batch_size):
                is_done = beam[b].advance(log_probs.data[b], copy_indices[b])
                if is_done:
                    done += [b]
                update_state(dec_hidden, b, beam[b].get_current_origin(), beam_size)
            if len(done) == batch_size:
                break

        # (4) back trace and find hyps
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[0]]
            k = ks[0]
            hyp = beam[b].get_hyp(k)
            all_hyp += [hyp]
        
        return all_hyp

def print_norm(t, name=''):
    """
    Print the L2 norm of a tensor.
    """
    n = torch.norm(t, p=2)
    print("L2-norm of {} is {}".format(name, n.data.cpu().numpy()))


