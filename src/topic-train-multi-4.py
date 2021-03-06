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
from vdcnn_multi_4 import MultiVDCNN


def lossfun(x, t):
    with cuda.get_device(t):
        soft20_x = x[:,:20]
        soft100_x = x[:,20:120]
        soft200_x = x[:,120:320]
        soft10n_x = x[:,320:]
        soft20_label = t[:,1:21]
        soft100_label = t[:,21:121]
        soft200_label = t[:,121:321]
        soft10n_label = t[:,321:]

    soft20_loss = - (F.sum(soft20_label * F.log_softmax(soft20_x))) / soft20_x.shape[0]
    soft100_loss = - (F.sum(soft100_label * F.log_softmax(soft100_x))) / soft100_x.shape[0]
    soft200_loss = - (F.sum(soft200_label * F.log_softmax(soft200_x))) / soft200_x.shape[0]
    soft10n_loss = - (F.sum(soft10n_label * F.log_softmax(soft10n_x))) / soft10n_x.shape[0]
    loss = soft20_loss + soft100_loss + soft200_loss + soft10n_loss
    return loss

def accfun(x, t):
    with cuda.get_device(t):
        label = t[:,0].astype('int32')
        soft100_x = x[:,20:120]
    return F.accuracy(soft100_x, label)


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
    train_df = pd.read_csv('../data/ag_news/topic100_train.csv')
    test_df = pd.read_csv('../data/ag_news/topic100_test.csv')
    drop_header = ['class_id', 'title', 'description']
    train_soft20 = pd.read_csv('../data/ag_news/topic20_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft20 = pd.read_csv('../data/ag_news/topic20_test.csv').drop(drop_header, axis=1).values.astype(np.float32)
    train_soft200 = pd.read_csv('../data/ag_news/topic200_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft200 = pd.read_csv('../data/ag_news/topic200_test.csv').drop(drop_header, axis=1).values.astype(np.float32)
    train_soft10n = pd.read_csv('../data/ag_news/topic40_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft10n = pd.read_csv('../data/ag_news/topic40_test.csv').drop(drop_header, axis=1).values.astype(np.float32)

elif mode == 'db_pedia':
    train_df = pd.read_csv('../data/db_pedia/topic100_train.csv')
    test_df = pd.read_csv('../data/db_pedia/topic100_test.csv')
    drop_header = ['class_id', 'title', 'description']
    train_soft20 = pd.read_csv('../data/db_pedia/topic20_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft20 = pd.read_csv('../data/db_pedia/topic20_test.csv').drop(drop_header, axis=1).values.astype(np.float32)
    train_soft200 = pd.read_csv('../data/db_pedia/topic200_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft200 = pd.read_csv('../data/db_pedia/topic200_test.csv').drop(drop_header, axis=1).values.astype(np.float32)
    train_soft10n = pd.read_csv('../data/db_pedia/topic140_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft10n = pd.read_csv('../data/db_pedia/topic140_test.csv').drop(drop_header, axis=1).values.astype(np.float32)

elif mode == 'yelpf':
    train_df = pd.read_csv('../data/yelp_review_full/topic100_train.csv')
    test_df = pd.read_csv('../data/yelp_review_full/topic100_test.csv')
    drop_header = ['class_id', 'review']
    train_soft20 = pd.read_csv('../data/yelp_review_full/topic20_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft20 = pd.read_csv('../data/yelp_review_full/topic20_test.csv').drop(drop_header, axis=1).values.astype(np.float32)
    train_soft200 = pd.read_csv('../data/yelp_review_full/topic200_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft200 = pd.read_csv('../data/yelp_review_full/topic200_test.csv').drop(drop_header, axis=1).values.astype(np.float32)
    train_soft10n = pd.read_csv('../data/yelp_review_full/topic50_train.csv').drop(drop_header, axis=1).values.astype(np.float32)
    test_soft10n = pd.read_csv('../data/yelp_review_full/topic50_test.csv').drop(drop_header, axis=1).values.astype(np.float32)


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

if mode == 'ag_news' or mode == 'db_pedia':
    drop_header = ['class_id', 'title', 'description']
elif mode == 'yelpf':
    drop_header = ['class_id', 'review']

train_y = train_df.drop(drop_header, axis=1).values.argmax(axis=1).astype(np.int32)
test_y = test_df.drop(drop_header, axis=1).values.argmax(axis=1).astype(np.int32)
train_soft = train_df.drop(drop_header, axis=1).values.astype(np.float32)
test_soft = test_df.drop(drop_header, axis=1).values.astype(np.float32)

n_out1 = train_soft20.shape[1]
n_out2 = train_soft.shape[1]
n_out3 = train_soft200.shape[1]
n_out4 = train_soft10n.shape[1]

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

print('MultiVDCNN setting: emb_dim={} n_out_1={}, n_out_2={}, n_out_3={}, n_out_4={}, depth={}'.format(len(char2id)+1, n_out1, n_out2, n_out3, n_out4, sum(depth)*2+1))

gpu_id = args.gpu

model = MultiVDCNN(len(char2id)+1, n_out1, n_out2, n_out3, n_out4, depth)
if gpu_id >= 0:
    model.to_gpu(gpu_id)


train_y = np.concatenate([train_y.reshape(-1, 1), train_soft20, train_soft, train_soft200, train_soft10n], axis=1).astype(np.float32)
test_y = np.concatenate([test_y.reshape(-1, 1), test_soft20, test_soft, test_soft200, test_soft10n], axis=1).astype(np.float32)

print(mode, train_x.shape, train_y.shape, test_x.shape, test_y.shape)

train = datasets.TupleDataset(train_x, train_y)

test = datasets.TupleDataset(test_x, test_y)

batch_size = 128

train_iter = iterators.SerialIterator(train, batch_size)
test_iter = iterators.SerialIterator(test, batch_size, False, False)

epoch_size = 5000
max_epoch = 15

model = L.Classifier(model, lossfun=lossfun, accfun=accfun)

optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)

optimizer.setup(model)

updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

result_dir = '../results/multi4_softtopic_{}_{}_{}_{}_{}_depth{}_valid{}'.format(mode, n_out1, n_out2, n_out3, n_out4, sum(depth)*2+1, args.valid)
trainer = training.Trainer(updater, (epoch_size * max_epoch, 'iteration'), out=result_dir)

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
