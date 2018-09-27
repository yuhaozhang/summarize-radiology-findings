"""
A trainer class to handle training and testing of models.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

from model.copy_model import Seq2SeqWithCopyModel
from model.loss import SequenceLoss, CoverageSequenceLoss
from utils import constant, torch_utils, text_utils

def unpack_batch(batch, opt):
    """ Unpack a batch from the data loader. """
    if opt['cuda']:
        inputs = [Variable(b.cuda()) if b is not None else None for b in batch[:4]]
    else:
        inputs = [Variable(b) if b is not None else None for b in batch[:4]]
    src_tokens = batch[4]
    tgt_tokens = batch[5]
    orig_idx = batch[6]
    return inputs, src_tokens, tgt_tokens, orig_idx

class Trainer(object):
    """ A trainer for training models. """
    def __init__(self, opt, vocab, emb_matrix=None):
        self.opt = opt
        # use pointer-generator
        self.model = Seq2SeqWithCopyModel(opt, emb_matrix=emb_matrix)
        # by default use 0 weight for coverage loss
        self.crit = CoverageSequenceLoss(vocab.size, opt.get('cov_alpha', 0))
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.crit.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.vocab = vocab

    def step(self, batch, eval=False):
        inputs, src_tokens, tgt_tokens, orig_idx = unpack_batch(batch, self.opt)
        src, tgt_in, tgt_out, bg = inputs
        
        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        log_probs, attn, cov = self.model(src, tgt_in, bg)
        loss = self.crit(log_probs.view(-1, self.vocab.size), tgt_out.view(-1), attn, cov)
        loss_val = loss.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, src_tokens, tgt_tokens, orig_idx = unpack_batch(batch, self.opt)
        src, _, _, bg = inputs
        
        self.model.eval()
        batch_size = src.size(0)
        preds = self.model.predict(src, bg, self.opt['beam_size'])
        # unmap words with copy
        pred_tokens = text_utils.unmap_with_copy(preds, src_tokens, self.vocab)
        pred_tokens = text_utils.prune_decoded_seqs(pred_tokens)
        if unsort:
            pred_tokens = text_utils.unsort(pred_tokens, orig_idx)
        return pred_tokens

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")
    
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

