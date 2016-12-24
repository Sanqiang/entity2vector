import os.path
import pickle
import random as rd
import sys
from collections import defaultdict, Counter, deque
from math import sqrt

import numpy as np
#import tensorflow as tf
from nltk.tokenize import TweetTokenizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import dok_matrix
from stemmer import PorterStemmer
from gensim.models import Word2Vec
import json
from collections import OrderedDict
from w2v_base import W2V_base
from w2v_cpp2 import W2V_cpp2

class Exp(W2V_base):

    def __init__(self, k, path, folder):
        self.k = k
        W2V_base.__init__(self, path, folder)
        self.train_path = "/".join((self.folder, "train.json"))
        self.test_path = "/".join((self.folder, "test.json"))

    def split(self):
        import json
        import heapq
        h = []
        with open(self.path_review, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                #reviewText = obj["reviewText"]
                #summary = obj["summary"]
                #reviewerID = obj["reviewerID"]
                #overall = obj["overall"]
                #asin = obj["asin"]
                unixReviewTime = int(obj["unixReviewTime"])
                heapq.heappush(h, unixReviewTime)
        split = int(len(h) * 0.8)

        f_train = open(self.train_path)
        f_test = open(self.test_path)
        with open(self.path_review, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                unixReviewTime = int(obj["unixReviewTime"])
                if unixReviewTime < split: # go to train
                    f_train.write(line)
                else:
                    f_test.write(line)
        return split

    def populate_score(self):
        matrix = dok_matrix((len(self.user2idx), len(self.prod2idx)), dtype=np.float)
        with open(self.train_path, "r", encoding=self.file_encoding) as ins:
            for line in ins:
                title, text, user, prod, rating = self.line_parser(line)
                user_idx = self.user2idx[user]
                prod_idx = self.prod2idx[prod]
                matrix[user_idx, prod_idx] = rating
        return matrix

    def populate_entity(self, path_vec,path_entity, prod_model = True):
        self.prod_model = prod_model
        self.entity_model = Word2Vec.load_word2vec_format(path_vec)

        self.entity2idx = {}
        self.idx2entity = OrderedDict()

        f = open(path_entity, "r")
        for line in f:
            entity = line[0:line.rindex("_")]
            idx = int(line[1+line.rindex("_"):])

            self.entity2idx[entity] = idx
            self.idx2entity[idx] = entity

    def predict(self, product_idx, user_idx):
        #find similar prod (which user already rate)
        inds = (self.matrix[:,user_idx] > 0).indices
        inds =[self.entity2idx[self.idx2prod[ind]] for ind in inds]
        list = self.entity_model.most_similar(self.entity2idx[self.idx2prod[product_idx]], topn=self.k, restrict_vocab=inds)

        preidct_score = 0
        denom = 0
        for pair in list:
            pair_ent = pair[0]
            prod = pair_ent[0:pair_ent.rindex("_")]
            prod_id = pair_ent[1+pair_ent.rindex("_"):]
            dist = pair[1]
            rating = self.matrix[prod_id, user_idx]
            preidct_score += rating * dist
            denom += dist
        preidct_score /= denom
        return preidct_score

    def test(self):
        rmse = []
        self.matrix = self.populate_score(self.train_path)
        self.tmatrix = self.populate_score(self.test_path)
        for prod_idx, user_idx in self.tmatrix.keys():
            truth_score = self.tmatrix[prod_idx, user_idx]
            preidct_score = self.predict(prod_idx, user_idx)
            rmse.append((truth_score - preidct_score)**2)
        print(np.mean(rmse))




if __name__ == '__main__':
    exp = Exp(10, "/Users/zhaosanqiang916/data/yelp/review_rest.json", "yelp_rest_prod")
    print("done init")
    exp.split()

    exp.populate_score()
    exp.populate_entity("/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod/output/model_88","/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod/prod.txt", prod_model=True)

    exp.test()