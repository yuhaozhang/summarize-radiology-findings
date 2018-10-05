"""
Run a summarization model interactively.
"""

import argparse
import numpy as np
import nltk
from colors import yellow, red, blue
import torch
from torch.autograd import Variable

from data import loader
from data.loader import DataLoader
from utils import helper, constant, torch_utils, text_utils, bleu, rouge
from utils.torch_utils import set_cuda
from utils.vocab import Vocab
from model.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Filename of the trained model .pt file.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # load model
    model_file = args.model_file
    print("Loading model from {}...".format(model_file))
    trainer = Trainer(model_file=model_file)
    opt, vocab = trainer.opt, trainer.vocab
    trainer.model.eval()
    print("Loaded.\n")

    # run
    while True:
        background, findings = get_input(opt)
        sum_words = run(background, findings, trainer, vocab, opt)
        print(blue("Predicted Impression:\n") + " ".join(sum_words))
        print("")
        inp = input("Quit (q to quit, otherwise continue)? ")
        if inp == 'q':
            break
    return

def get_input(opt):
    while True:
        print("")
        background = input(yellow("Input the Background Section (<= 100 words or leave empty):\n> "))
        background = nltk.word_tokenize(background)
        if len(background) > 100:
            print(red("Too long, please keep it within 100 words."))
        else:
            break
    while True:
        print("")
        findings = input(yellow("Input the Findings Section (<= 500 words):\n> "))
        findings = nltk.word_tokenize(findings)
        if len(findings) > 500:
            print(red("Too long, please keep it within 500 words."))
        elif len(findings) < 2:
            print(red("Too short."))
        else:
            break
    print("")
    return background, findings

def run(background, findings, trainer, vocab, opt):
    # preprocess data
    bg_tokens, src_tokens = background, findings
    if opt['lower']:
        bg_tokens = [t.lower() for t in bg_tokens]
        src_tokens = [t.lower() for t in src_tokens]
    if len(bg_tokens) == 0:
        bg_tokens = [constant.UNK_TOKEN]
    src_tokens = [constant.SOS_TOKEN] + src_tokens + [constant.EOS_TOKEN]
    src = loader.map_to_ids(src_tokens, vocab.word2id)
    bg = loader.map_to_ids(bg_tokens, vocab.word2id)
    src = loader.get_long_tensor([src], 1)
    bg = loader.get_long_tensor([bg], 1)
    if opt['cuda']:
        src = src.cuda()
        bg = bg.cuda()
    
    preds = trainer.model.predict(src, bg, opt['beam_size'])
    pred_tokens = text_utils.unmap_with_copy(preds, [src_tokens], vocab)
    pred_tokens = text_utils.prune_decoded_seqs(pred_tokens)[0]
    return pred_tokens

if __name__ == '__main__':
    main()
