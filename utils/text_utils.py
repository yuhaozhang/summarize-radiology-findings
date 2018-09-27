"""
Text specific utility functions.
"""
from collections import Counter
import random
import unicodedata

from utils import constant

def postprocess(preds):
    """ Postprocess summaries using rules. """
    processed = []
    for ps in preds:
        new = []
        if len(ps) > 2 and ps[-1] != '.' and ps[-2] == '.':
            ps = ps[:-1]
        for i, p in enumerate(ps):
            if i > 0 and ps[i-1] == p:
                continue
            new += [p]
        processed += [new]
    return processed

def save_predictions(preds, filename):
    with open(filename, 'w') as outfile:
        for tokens in preds:
            print(' '.join(tokens), file=outfile)
    print("Predictions saved to file: " + filename)

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def unmap_with_copy(indices, src_tokens, vocab):
    """
    Unmap a list of list of indices, by optionally copying from src_tokens.
    """
    result = []
    for ind, tokens in zip(indices, src_tokens):
        words = []
        for idx in ind:
            if idx >= 0:
                words.append(vocab.id2word[idx])
            else:
                idx = -idx - 1 # flip and minus 1
                words.append(tokens[idx])
        result += [words]
    return result

def prune_decoded_seqs(seqs):
    """
    Prune decoded sequences after EOS token.
    """
    out = []
    for s in seqs:
        if constant.EOS_TOKEN in s:
            idx = s.index(constant.EOS_TOKEN)
            out += [s[:idx]]
        else:
            out += [s]
    return out

def prune_hyp(hyp):
    """
    Prune a decoded hypothesis
    """
    if constant.EOS_ID in hyp:
        idx = hyp.index(constant.EOS_ID)
        return hyp[:idx]
    else:
        return hyp

def prune(data_list, lens):
    assert len(data_list) == len(lens)
    nl = []
    for d, l in zip(data_list, lens):
        nl.append(d[:l])
    return nl

def sort(packed, ref, reverse=True):
    """
    Sort a series of packed list, according to a ref list.
    Also return the original index before the sort.
    """
    assert (isinstance(packed, tuple) or isinstance(packed, list)) and isinstance(ref, list)
    packed = [ref] + [range(len(ref))] + list(packed)
    sorted_packed = [list(t) for t in zip(*sorted(zip(*packed), reverse=reverse))]
    return tuple(sorted_packed[1:])

def unsort(sorted_list, oidx):
    """
    Unsort a sorted list, based on the original idx.
    """
    assert len(sorted_list) == len(oidx), "Number of list elements must match with original indices."
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted

def map_to_idx(data_list, vocab):
    """
    Map a list of tokens to idx.
    """
    if isinstance(data_list[0][0], list):
        # is a list of list of list
        data_by_idx = []
        for d in data_list:
            data_by_idx.append(list(map(vocab.map, d)))
        return data_by_idx
    # is a list of list
    data_by_idx = list(map(vocab.map, data_list))
    return data_by_idx

def lowercase_data(data):
    """
    Lowercase all tokens.
    """
    new_data = []
    for l in data:
        if isinstance(l[0], list):
            # list of list
            new_l = lowercase_data(l)
        else:
            new_l = [t.lower() for t in l]
        new_data.append(new_l)
    return new_data

def get_words(texts):
    """
    Construct a word counter from words.
    """
    word_counter = Counter()
    for t in texts:
        word_counter.update(t)
    return word_counter

def shuffle_fields(fields):
    """
    Shuffle all fields in a dict together. Each field should be a list.
    """
    keys, values = zip(*fields.items())
    zipped = list(zip(*values))
    random.shuffle(zipped)
    unzipped = list(zip(*zipped))
    for k, v in zip(keys, unzipped):
        fields[k] = list(v)
    return fields

