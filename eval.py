"""
Evaluating saved models.
"""

import argparse
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable

from data.loader import DataLoader
from utils import helper, constant, torch_utils, text_utils, bleu, rouge
from utils.torch_utils import set_cuda
from utils.vocab import Vocab
from model.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Directory of the model file.')
parser.add_argument('--data_dir', default='', help='Directory to look for data. By default use the path in loaded args.')
parser.add_argument('--model', default='best_model.pt', help='Name of the model file.')
parser.add_argument('--dataset', default='test', help="Data split to use for evaluation: dev or test.")
parser.add_argument('--batch_size', type=int, default=100, help="Batch size for evaluation.")
parser.add_argument('--gold', default='', help="Optional: a file where to write gold summarizations. Default to not write.")
parser.add_argument('--out', default='', help="Optional: a file where to write predictions. Default to not write.")
parser.add_argument('--use_bleu', action='store_true', help="Use BLEU instead of ROUGE metrics for scoring.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
trainer = Trainer(model_file=model_file)
opt, vocab = trainer.opt, trainer.vocab

# load data
data_dir = args.data_dir if len(args.data_dir) > 0 else opt['data_dir']
data_file = data_dir + '/{}.jsonl'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, args.batch_size))
batch = DataLoader(data_file, args.batch_size, opt, vocab, evaluation=True)
test_gold = batch.save_gold(args.gold)

helper.print_config(opt)

print("Evaluating on the {} set...".format(args.dataset))
predictions = []
for b in tqdm(batch):
    preds = trainer.predict(b)
    predictions += preds
predictions = text_utils.postprocess(predictions)
if args.use_bleu:
    test_bleu = bleu.get_bleu(predictions, test_gold)
    print("{} set bleu score: {:.2f}".format(args.dataset, test_bleu))
else:
    r1, r2, rl, r1_cf, r2_cf, rl_cf = rouge.get_rouge(predictions, test_gold, use_cf=True)
    print("{} set results:\n".format(args.dataset))
    print("Metric\tScore\t95% CI")
    print("ROUGE-1\t{:.2f}\t({:.2f},{:.2f})".format(r1, r1_cf[0]-r1, r1_cf[1]-r1))
    print("ROUGE-2\t{:.2f}\t({:.2f},{:.2f})".format(r2, r2_cf[0]-r2, r2_cf[1]-r2))
    print("ROUGE-L\t{:.2f}\t({:.2f},{:.2f})".format(rl, rl_cf[0]-rl, rl_cf[1]-rl))

if len(args.out) > 0:
    text_utils.save_predictions(predictions, args.out)

