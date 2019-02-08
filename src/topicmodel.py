import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import lda
from sklearn.externals import joblib
import time

parser = argparse.ArgumentParser(description='Train from scratch.')
parser.add_argument('--mode', '-m', dest='mode',default='ag_news',
                    help='use dataset (ag_news, db_pedia, yelpf)')
parser.add_argument('--num', '-n', dest='n_topics', default=100, type=int,
                    help='n_topics')
parser.add_argument('--iter', '-i', dest='n_iter', default=2000, type=int,
                    help='n_iter')

args = parser.parse_args()

mode = args.mode
print('mode:{}'.format(mode))
if mode == 'ag_news' or mode == 'db_pedia':
    header = ['class_id', 'title', 'description']
    train_df = pd.read_csv('../data/{}/train.csv'.format(mode), header=None, names=header)
    train_df.title = train_df.title.str.lower()
    train_df.description = train_df.description.str.lower()
    train_x = train_df.title.values + ' ' + train_df.description.values 
elif mode == 'yelpf':
    header = ['class_id', 'review']
    train_df = pd.read_csv('../data/yelp_review_full/train.csv', header=None, names=header)
    train_df.review = train_df.review.str.lower()
    train_x = train_df.review.values

if mode == 'ag_news':
    bow_model = joblib.load('../results/lda/models/bow_model.pkl')
elif mode == 'db_pedia':
    bow_model = joblib.load('../results/lda/models/dbpedia_bow_model.pkl')
elif mode == 'yelpf':
    bow_model = joblib.load('../results/lda/models/yelpf_bow_model.pkl')

bow = bow_model.transform(train_x)

n_topics = args.n_topics
n_iter = args.n_iter
print('n_topics:{}, n_iter:{}'.format(n_topics, n_iter))

with open('../results/lda/log/{}_{}_iter{}_done_time.txt'.format(mode, n_topics, n_iter), 'w') as f:
    start = time.time()
    lda_model = lda.lda.LDA(n_topics=n_topics, n_iter=n_iter, random_state=0)
    lda_model.fit(bow)
    joblib.dump(lda_model, '../results/lda/models/{}_lda_model_{}_{}iter.pkl'.format(mode, n_topics, n_iter))
    end = time.time()
    print("topic_N =", str(n_topics), "train time", end - start, file=f)
    