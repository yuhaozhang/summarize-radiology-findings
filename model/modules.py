"""
Pytorch implementation of basic sequence to Sequence models.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np

from utils import constant

class BasicAttention(nn.Module):
    """
    A basic MLP attention layer.
    """
    def __init__(self, dim):
        super(BasicAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input) # batch x dim
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        attn = target.unsqueeze(1).expand_as(context) + source
        attn = self.tanh(attn) # batch x sourceL x dim
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)

        if mask is not None:
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), dim=1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, weighted_context, attn

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), dim=1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, weighted_context, attn


class LinearAttention(nn.Module):
    """ A linear attention form, inspired by BiDAF:
        a = W (u; v; u o v)
    """

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.linear = nn.Linear(dim*3, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)  # batch*sourceL x dim
        v = context.contiguous().view(-1, dim)
        attn_in = torch.cat((u, v, u.mul(v)), 1)
        attn = self.linear(attn_in).view(batch_size, source_len)

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(batch_size, 1, source_len)  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), dim=1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, weighted_context, attn

class DeepAttention(nn.Module):
    """ A deep attention form, invented by Robert:
        u = ReLU(Wx)
        v = ReLU(Wy)
        a = V.(u o v)
    """

    def __init__(self, dim):
        super(DeepAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)  # batch*sourceL x dim
        u = self.relu(self.linear_in(u))
        v = self.relu(self.linear_in(context.contiguous().view(-1, dim)))
        attn = self.linear_v(u.mul(v)).view(batch_size, source_len)

        if mask is not None:
            # sett the padding attention logits to -inf
            assert mask.size() == attn.size(), "Mask size must match the attention size!"
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        attn3 = attn.view(batch_size, 1, source_len)  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), dim=1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, weighted_context, attn

class CoverageAttention(nn.Module):
    """
    A MLP attention layer with coverage vector in it.
    """
    def __init__(self, dim):
        super(CoverageAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_cov = nn.Linear(1, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)
        print("[Using coverage in attention layer.]")

    def forward(self, input, context, coverage, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        coverage: batch x sourceL
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input) # batch x dim
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        cov = self.linear_cov(coverage.unsqueeze(2))
        attn = target.unsqueeze(1).expand_as(context) + source + cov
        attn = self.tanh(attn) # batch x sourceL x dim
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)

        if mask is not None:
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)

        attn = self.sm(attn)
        if attn_only:
            return attn

        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), dim=1)
        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, weighted_context, attn


class LSTMAttention(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, attn_type='soft', coverage=False):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.coverage = coverage

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        
        if coverage: # first see if coverage is applied
            self.attention_layer = CoverageAttention(hidden_size)
        else:
            if attn_type == 'soft':
                self.attention_layer = SoftDotAttention(hidden_size)
            elif attn_type == 'mlp':
                self.attention_layer = BasicAttention(hidden_size)
            elif attn_type == 'linear':
                self.attention_layer = LinearAttention(hidden_size)
            elif attn_type == 'deep':
                self.attention_layer = DeepAttention(hidden_size)
            else:
                raise Exception("Unsupported LSTM attention type: {}".format(attn_type))
            print("Using {} attention for LSTM.".format(attn_type))

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # n_b x hidden_dim

            return hy, cy
        
        input = input.transpose(0,1) # n_step x n_b x hidden_dim
        nsteps, nbatch = input.size(0), input.size(1)
        ctx_len = ctx.size(1)
        
        h_out, h_tilde_out, c_out, attn_out = [], [], [], []
        cov_out = []

        steps = range(nsteps)
        attn = torch.zeros([nbatch, ctx_len]).cuda() # initial attn, for coverage
        cov = 0
        for i in steps:
            # advance rnn
            hidden = recurrence(input[i], hidden)
            hy, cy = hidden

            # do attention
            if self.coverage:
                cov = cov + attn
                cov_out.append(cov)
                h_tilde, c, attn = self.attention_layer(hy, ctx, cov, mask=ctx_mask)
            else:
                h_tilde, c, attn = self.attention_layer(hy, ctx, mask=ctx_mask)

            # save results
            h_out.append(hy)
            h_tilde_out.append(h_tilde)
            c_out.append(c)
            attn_out.append(attn)
        h_out = torch.cat(h_out, dim=0).view(nsteps, *h_out[0].size())
        h_tilde_out = torch.cat(h_tilde_out, dim=0).view(nsteps, *h_tilde_out[0].size())
        c_out = torch.cat(c_out, dim=0).view(nsteps, *c_out[0].size())
        attn_out = torch.cat(attn_out, dim=0).view(nsteps, *attn_out[0].size())

        for x in [h_out, h_tilde_out, c_out, attn_out]:
            x.transpose_(0,1)

        if self.coverage:
            cov_out = torch.cat(cov_out, dim=0).view(nsteps, *cov_out[0].size())
            cov_out.transpose_(0,1)

        return h_out, h_tilde_out, c_out, attn_out, cov_out, hidden

