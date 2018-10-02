"""
Different loss functions.
"""

import torch
import torch.nn as nn

from utils import constant

def SequenceLoss(vocab_size):
    weight = torch.ones(vocab_size)
    weight[constant.PAD_ID] = 0
    crit = nn.NLLLoss(weight)
    print("Using NLL sequence loss.")
    return crit

class CoverageSequenceLoss(nn.Module):
    """
    A sequence NLL loss wrapper that also supports additional coverage loss.

    Loss = NLLLoss + alpha * CoverageLoss
    """
    def __init__(self, vocab_size, alpha):
        super().__init__()
        weight = torch.ones(vocab_size)
        weight[constant.PAD_ID] = 0
        self.nll = nn.NLLLoss(weight)
        self.alpha = alpha

    def forward(self, inputs, targets, attn, cov):
        assert inputs.size(0) == targets.size(0)
        nll_loss = self.nll(inputs, targets)
        if self.alpha == 0:
            return nll_loss
        # add coverage loss
        cov_loss = torch.sum(torch.min(attn, cov), dim=2).view(-1)
        pad_mask = targets.eq(constant.PAD_ID)
        unpad_mask = targets.ne(constant.PAD_ID)
        cov_loss.masked_fill_(pad_mask, 0)
        denom = torch.sum(unpad_mask).float()
        cov_loss = torch.sum(cov_loss) / (denom + constant.SMALL_NUMBER)
        return nll_loss + self.alpha * cov_loss

    def update_alpha(self, alpha):
        print("[Update coverage loss weight to be {}]".format(alpha))
        self.alpha = alpha

class MaxEntropySequenceLoss(nn.Module):
    """
    A max entropy loss that encourage the model to have large entropy,
    therefore giving more diverse outputs.

    Loss = NLLLoss + alpha * EntropyLoss
    """
    def __init__(self, vocab_size, alpha):
        super().__init__()
        weight = torch.ones(vocab_size)
        weight[constant.PAD_ID] = 0
        self.nll = nn.NLLLoss(weight)
        self.alpha = alpha
        print("Using max entropy sequence loss.")

    def forward(self, inputs, targets):
        """
        inputs: [N, C]
        targets: [N]
        """
        assert inputs.size(0) == targets.size(0)
        nll_loss = self.nll(inputs, targets)
        # entropy loss
        mask = targets.eq(constant.PAD_ID).unsqueeze(1).expand_as(inputs)
        masked_inputs = inputs.clone().masked_fill_(mask, 0.0)
        p = torch.exp(masked_inputs)
        ent_loss = p.mul(masked_inputs).sum() / inputs.size(0) # average over minibatch
        loss = nll_loss + self.alpha * ent_loss
        return loss

