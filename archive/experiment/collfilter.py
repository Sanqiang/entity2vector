import os.path
import pickle
import random as rd
import sys
from collections import defaultdict, Counter, deque
from math import sqrt
import math

import numpy as np
#import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import dok_matrix
#from stemmer import PorterStemmer
from gensim.models import Word2Vec
import json
from collections import OrderedDict
import time
import datetime

home = os.environ["HOME"]

class Exp():

    def __init__(self, k = 10):
        self.k = k
        self.data_path = "".join((home, "/data/yelp/review_rest.json"))
        self.train_path = "train80p.json"
        self.test_path = "test80p.json"
        self.prod_vector = "".join((home, "/data/model/prod_sg_72.896439.prod.vec"))
        self.prod2idx = {}
        self.idx2prod = []
        self.user2idx = {}
        self.idx2user = []
        self.word_cnt = 210185

    def split(self):
        import json
        import heapq
        h = []

        with open(self.data_path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                date = obj["date"]
                unixReviewTime = time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple())
                heapq.heappush(h, unixReviewTime)
        split = h[int(len(h) * 0.8)]

        f_train = open(self.train_path, "w")
        f_test = open(self.test_path, "w")
        train_cnt = 0
        test_cnt = 0
        with open(self.data_path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                date = obj["date"]
                unixReviewTime = time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple())
                if unixReviewTime < split: # go to train
                    f_train.write(line)
                    ++train_cnt
                else:
                    f_test.write(line)
                    ++test_cnt
        print(train_cnt, test_cnt)
        return split

    def initlize(self):
        #process syn0 with prod2idx
        f = open(self.prod_vector, "r")
        line = f.readline()
        vocab_size, self.vector_size = map(int, line.split())
        self.syn0 = np.zeros((vocab_size, self.vector_size), dtype=float)

        def add_prod(prod, weights):
            if prod not in self.prod2idx:
                #process word2idx & idx2word
                self.prod2idx[prod] = len(self.idx2prod)
                self.idx2prod.append(prod)
                #process syn0
                self.syn0[self.prod2idx[prod]] = weights

        for line_no, line in enumerate(f):
            parts = line.split()
            if len(parts) != self.vector_size+1:
                raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
            prod, weights = parts[0], list(map(float, parts[1:]))
            add_prod(prod, weights)

        #process user2idx
        with open(self.data_path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                user_id = obj["user_id"]
                if user_id not in self.user2idx:
                    self.user2idx[user_id] = len(self.idx2user)
                    self.idx2user.append(user_id)

        #process rating matrix
        self.train_score_matrix = dok_matrix((len(self.user2idx), len(self.prod2idx)), dtype=np.float)
        with open(self.train_path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                user_id = obj["user_id"]
                review_id = obj["review_id"]
                business_id = obj["business_id"]
                stars = obj["stars"]
                user_idx = self.user2idx[user_id]
                prod_idx = self.prod2idx[business_id]
                self.train_score_matrix[user_idx, prod_idx] = stars

        self.test_score_matrix = dok_matrix((len(self.user2idx), len(self.prod2idx)), dtype=np.float)
        with open(self.test_path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                user_id = obj["user_id"]
                review_id = obj["review_id"]
                business_id = obj["business_id"]
                stars = obj["stars"]
                user_idx = self.user2idx[user_id]
                prod_idx = self.prod2idx[business_id]

                self.test_score_matrix[user_idx, prod_idx] = stars

    def most_similar(self, query_prod_id, restrict_inds = None, k = 10):
        if restrict_inds is None:
            dists = np.dot(self.syn0[query_prod_id,], self.syn0.T)
        else:
            dists = np.dot(self.syn0[query_prod_id,], self.syn0[restrict_inds,].T)
        best_idxs = np.argsort(dists)[::-1][:k]
        best = [restrict_inds[best_idx] for best_idx in best_idxs]
        return zip(best,dists[best_idxs])

    def predict(self, product_idx, user_idx):
        #find similar prod (which user already rate)
        inds = (self.train_score_matrix[user_idx,:] > 0).indices
        if len(inds) == 0:
            return -1
        pairs = self.most_similar(product_idx, restrict_inds=inds)

        preidct_score = 0
        denom = 0
        for prod_id,dist in pairs:
            rating = self.train_score_matrix[user_idx, prod_id]
            preidct_score += rating * dist
            denom += dist
        preidct_score /= denom
        if math.isnan(preidct_score):
            return -1
        return preidct_score

    def test(self):
        self.initlize()
        rmse = []
        for user_idx,prod_idx in self.test_score_matrix.keys():
            truth_score = self.test_score_matrix[user_idx, prod_idx]
            preidct_score = self.predict(prod_idx, user_idx)
            if preidct_score == -1:
                continue
            rmse.append((truth_score - preidct_score)**2)
        print(math.sqrt(np.mean(rmse)))




if __name__ == '__main__':
    exp = Exp()
    # exp.split()
    exp.initlize()
    exp.test()
