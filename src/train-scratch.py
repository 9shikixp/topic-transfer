import argparse
from collections import Counter
import pandas as pd

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import os, sys
sys.path.append(os.pardir+'/src')
from vdcnn import VDCNN


parser = argparse.ArgumentParser(description='Train from scratch.')
parser.add_argument('--mode', '-m', dest='mode',default='ag_news',
                    help='use dataset (ag_news, db_pedia, yelpf)')
parser.add_argument('--gpu', '-g', dest='gpu', default=0, type=int,
                    help='gpu_id (0 or 1)')
parser.add_argument('--depth', '-d', dest='depth', default=17, type=int,
                    help='conv depth (9, 17, 29)')
parser.add_argument('--valid', '-v', dest='valid', default=0, type=int,
                    help='valid id')

args = parser.parse_args()


mode = args.mode

if mode == 'ag_news':
    header = ['class_id', 'title', 'description']
    train_df = pd.read_csv('../data/ag_news/train.csv', header=None, names=header)
    test_df = pd.read_csv('../data/ag_news/test.csv', header=None, names=header)
elif mode == 'db_pedia':
    header = ['class_id', 'title', 'description']
    train_df = pd.read_csv('../data/db_pedia/train.csv', header=None, names=header)
    test_df = pd.read_csv('../data/db_pedia/test.csv', header=None, names=header)
elif mode == 'yelpf':
    header = ['class_id', 'review']
    train_df = pd.read_csv('../data/yelp_review_full/train.csv', header=None, names=header)
    test_df = pd.read_csv('../data/yelp_review_full/test.csv', header=None, names=header)


if mode == 'ag_news' or mode == 'db_pedia':
    train_df.title = train_df.title.str.lower()
    train_df.description = train_df.description.str.lower()
    test_df.title = test_df.title.str.lower()
    test_df.description = test_df.description.str.lower()
elif mode == 'yelpf':
    train_df.review = train_df.review.str.lower()
    test_df.review = test_df.review.str.lower()


characters = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/|_#$%^&*~`+=<>()[]{}\t "
char2id = {}
for i, c in enumerate(characters):
    char2id[c] = i


if mode == 'ag_news' or mode == 'db_pedia':
    train_x = train_df.title.values + ' ' + train_df.description.values
    test_x = test_df.title.values + ' ' + test_df.description.values
elif mode == 'yelpf':
    train_x = train_df.review.values
    test_x = test_df.review.values

train_x = [[char2id[c] if c in char2id else len(char2id) for c in s][:1024] for s in train_x]
test_x = [[char2id[c] if c in char2id else len(char2id) for c in s][:1024] for s in test_x]

f = lambda x: np.pad(x, pad_width=(0, 1024-len(x)), mode='constant', constant_values=-1)
train_x = np.array(list(map(f, train_x)), np.int32)
test_x = np.array(list(map(f, test_x)), np.int32)

train_y = train_df.class_id.values.astype(np.int32) -1
test_y = test_df.class_id.values.astype(np.int32) -1

kind = len(Counter(train_y))

depth = args.depth
if depth == 9:
    depth = [1, 1, 1, 1]
elif depth == 17:
    depth = [2, 2, 2, 2]
elif depth == 29:
    depth = [5, 5, 2, 2]
else:
    print('depth must be (9, 17, 29)')
    sys.exit()

print('VDCNN setting: emb_dim={} n_out={}, depth={}'.format(len(char2id)+1, kind, sum(depth)*2+1))

gpu_id = args.gpu

model = VDCNN(len(char2id)+1, kind, depth)
if gpu_id >= 0:
    model.to_gpu(gpu_id)

print(mode, train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
train = datasets.TupleDataset(train_x, train_y)
test = datasets.TupleDataset(test_x, test_y)

batch_size = 128

train_iter = iterators.SerialIterator(train, batch_size)
test_iter = iterators.SerialIterator(test, batch_size, False, False)


epoch_size = 5000
max_epoch = 15

model = L.Classifier(model)

optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)

optimizer.setup(model)

updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

result_dir = '../results/scratch_{}_depth{}_valid{}'.format(mode, sum(depth)*2+1, args.valid)
trainer = training.Trainer(updater, (epoch_size * max_epoch, 'iteration'),
                           out=result_dir)

from chainer.training import extensions

trainer.extend(extensions.LogReport(trigger=(epoch_size, 'iteration')))
trainer.extend(extensions.snapshot(filename='snapshot_iteration-{.updater.iteration}'), trigger=(epoch_size, 'iteration'))
trainer.extend(extensions.snapshot_object(model.predictor, filename='model_iteration-{.updater.iteration}'), trigger=(epoch_size, 'iteration'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id), trigger=(epoch_size, 'iteration'))
trainer.extend(extensions.observe_lr(), trigger=(epoch_size, 'iteration'))
trainer.extend(extensions.PrintReport(['iteration', 'lr', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']), trigger=(epoch_size, 'iteration'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(epoch_size*3, 'iteration'))
trainer.extend(extensions.ProgressBar(update_interval=30))

print('running')
print('reslut_dir:{}'.format(result_dir))

trainer.run()
