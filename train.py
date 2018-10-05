"""
Train a summarization model.
By default, a seq2seq model with attention and copy mechanism is used.
"""

import os
import shutil
import time
from datetime import datetime
import argparse
import numpy as np
import random
import torch

from data.loader import DataLoader
from utils import helper, constant, torch_utils, text_utils, bleu, rouge
from utils.torch_utils import set_cuda
from utils.vocab import Vocab
from model.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/stanford-reports', help='Directory for jsonl data')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab', help='Directory for vocab')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--emb_dim', type=int, default=100, help='Input embedding size.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of encoding layers.')
parser.add_argument('--emb_dropout', type=float, default=0.5, help='Dropout rate for embedding vectors.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate used in encoder and decoder.')
parser.add_argument('--lower', action='store_true', help='Lowercase all input words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=True)
parser.add_argument('--max_dec_len', type=int, default=100, help='Max decoding length.')
parser.add_argument('--beam_size', type=int, default=5, help='Beam search size used in decoder.')
parser.add_argument('--top', type=int, default=1000000, help='Only tune top k embeddings; by default tune all.')
parser.add_argument('--train_data', default='train', help='Name of training file.')
parser.add_argument('--dev_data', default='dev', help='Name of dev file.')

parser.add_argument('--attn_type', choices=['mlp', 'soft', 'linear', 'deep'], default='mlp', help='Attention function. See model/modules.py for details.')
parser.add_argument('--cov', action='store_true', help='Use coverage mechanism in attention.')
parser.add_argument('--cov_alpha', type=float, default=0, help='Weight alpha for coverage loss.')
parser.add_argument('--cov_loss_epoch', type=int, default=0, help='Add coverage loss starting from this epoch.')

parser.add_argument('--background', action='store_true', help='Use background information for decoder.')
parser.add_argument('--concat_background', action='store_true', help='Simply concat background to findings.')

parser.add_argument('--use_bleu', action='store_true', help='Use BLEU as the metric. By default use ROUGE.')
parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=30, help='Decay the lr starting from this epoch.')
parser.add_argument('--optim', choices=['adam', 'sgd', 'adagrad', 'adamax'], default='adam', help='Optimizer (default to adam).')
parser.add_argument('--num_epoch', type=int, default=30, help='Total number of training epochs.')
parser.add_argument('--batch_size', type=int, default=25, help='Batch size for training.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

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

# make opt
opt = vars(args)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/{}.jsonl'.format(opt['train_data']), opt['batch_size'], opt, vocab, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/{}.jsonl'.format(opt['dev_data']), opt['batch_size'], opt, vocab, evaluation=True)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score")

# save gold predictions
train_gold = train_batch.save_gold(model_save_dir + '/pred_train_gold.txt')
dev_gold = dev_batch.save_gold(model_save_dir + '/pred_dev_gold.txt')

# print model info
helper.print_config(opt)

trainer = Trainer(opt=opt, vocab=vocab, emb_matrix=emb_matrix)
if opt['cov_loss_epoch'] > 1: # delay coverage loss
    trainer.crit.update_alpha(0)

global_step = 0
max_steps = len(train_batch) * opt['num_epoch']
dev_score_history = []
current_lr = opt['lr']
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

# start training
for epoch in range(1, opt['num_epoch']+1):
    if epoch >= opt['cov_loss_epoch']: # add coverage loss back
        trainer.crit.update_alpha(opt['cov_alpha'])

    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.step(batch, eval=False) # update step
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step,\
                    max_steps, epoch, opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    dev_preds = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        loss = trainer.step(batch, eval=True) # eval step
        dev_loss += loss
        preds = trainer.predict(batch)
        dev_preds += preds
    dev_preds = text_utils.postprocess(dev_preds)
    text_utils.save_predictions(dev_preds, model_save_dir + '/pred_dev_e{}.txt'.format(epoch))
    if opt['use_bleu']:
        dev_score = bleu.get_bleu(dev_preds, dev_gold)
    else:
        rouge1, rouge2, rougel = rouge.get_rouge(dev_preds, dev_gold, use_cf=False)
        dev_score = rougel
    
    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_score = {:.4f}".format(epoch,\
            train_loss, dev_loss, dev_score))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_score))

    # save
    model_file = model_save_dir + '/best_model.pt'
    if epoch == 1 or dev_score > max(dev_score_history):
        trainer.save(model_file, epoch)
        print("[new best model saved.]")
    
    # lr schedule
    if epoch > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")

print("Training ended with {} epochs.".format(epoch))


