"""
Data loader from report json data.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab, jsonl

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation

        with open(filename) as infile:
            data = jsonl.load(infile)

        # filter and sample data
        if opt.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(opt['sample_train'] * len(data))
            data = random.sample(data, keep)
            print("Subsample training set with rate {:g}".format(opt['sample_train']))

        self.raw_data = data
        data = self.preprocess(data, vocab, opt)
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            self.raw_data = [self.raw_data[i] for i in indices]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}.".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids using vocab. """
        processed = []
        assert opt.get('background', False) == False or opt.get('concat_background', False) == False,\
                "Cannot use background encoder when concat background."
        for d in data:
            # remove section headers (e.g., "FINDINGS :")
            bg_tokens = d['background']
            src_tokens = d['findings'][2:]
            tgt_tokens = d['impression'][2:]
            if opt['lower']:
                bg_tokens = [t.lower() for t in bg_tokens]
                src_tokens = [t.lower() for t in src_tokens]
                tgt_tokens = [t.lower() for t in tgt_tokens]
            if len(bg_tokens) == 0:
                bg_tokens = [constant.UNK_TOKEN] # use unk as a dummy background
            if opt.get('concat_background', False):
                src_tokens = bg_tokens + [constant.SEP_TOKEN] + src_tokens
            src_tokens = [constant.SOS_TOKEN] + src_tokens + [constant.EOS_TOKEN]
            tgt_in = [constant.SOS_TOKEN] + tgt_tokens # target fed in RNN
            tgt_out = tgt_tokens + [constant.EOS_TOKEN] # target from RNN output
            src = map_to_ids(src_tokens, vocab.word2id)
            tgt_in = map_to_ids(tgt_in, vocab.word2id)
            tgt_out = map_to_ids(tgt_out, vocab.word2id)
            bg = map_to_ids(bg_tokens, vocab.word2id)
            processed += [[src_tokens, tgt_tokens, src, tgt_in, tgt_out, bg]]
        return processed

    def save_gold(self, filename):
        gold = [d['impression'][2:] for d in self.raw_data]
        if self.opt['lower']:
            gold = [[t.lower() for t in g] for g in gold]
        if len(filename) > 0:
            with open(filename, 'w') as outfile:
                for seq in gold:
                    print(" ".join(seq), file=outfile)
        return gold

    def get_src(self):
        return [d['findings'][2:] for d in self.raw_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 6
        
        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # convert to tensors
        src_tokens = batch[0]
        tgt_tokens = batch[1]
        src = get_long_tensor(batch[2], batch_size)
        tgt_in = get_long_tensor(batch[3], batch_size)
        tgt_out = get_long_tensor(batch[4], batch_size)
        bg = get_long_tensor(batch[5], batch_size)
        assert tgt_in.size(1) == tgt_out.size(1), \
                "Target input and output sequence sizes do not match."
        return (src, tgt_in, tgt_out, bg, src_tokens, tgt_tokens, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_long_tensor(tokens_list, batch_size, pad_id=constant.PAD_ID):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(pad_id)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_float_tensor(features_list, batch_size):
    if features_list is None or features_list[0] is None:
        return None
    seq_len = max(len(x) for x in features_list)
    feature_len = len(features_list[0][0])
    features = torch.FloatTensor(batch_size, seq_len, feature_len).zero_()
    for i,f in enumerate(features_list):
        features[i,:len(f),:] = torch.FloatTensor(f)
    return features

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

