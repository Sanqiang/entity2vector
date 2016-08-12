import os.path
import pickle
import random as rd
import sys
from collections import defaultdict, Counter, deque
from math import sqrt

import numpy as np
import tensorflow as tf
from nltk.tokenize import TweetTokenizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from stemmer import PorterStemmer
from w2v_s import W2V_c
from gensim.models import Word2Vec
import json

class Exp:

    def __init__(self, k):
        self.k = k

    def split(self):
        import json
        import heapq
        h = []
        with open(self.path, "r") as ins:
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

        f_train = open("/".join((self.folder, "train.json")))
        f_test = open("/".join((self.folder, "test.json")))
        with open(self.path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                unixReviewTime = int(obj["unixReviewTime"])
                if unixReviewTime < split: # go to train
                    f_train.write(line)
                else:
                    f_test.write(line)

        return split

    def populate_idx(self):

        self.prod2idx ={}
        self.idx2prod = []
        path = "/".join((self.folder, "prod.txt"))
        with open(path, "r") as ins:
            for line in ins:
                prod = line[0:line.rindex("_")]
                self.prod2idx[prod] = len(self.prod2idx)
                self.idx2prod.append(prod)

        self.user2idx = {}
        self.idx2user = []
        path = "/".join((self.folder, "user.txt"))
        with open(path, "r") as ins:
            for line in ins:
                user = line[0:line.rindex("_")]
                self.user2idx[user] = len(self.user2idx)
                self.idx2user.append(user)

    def get_review_model(self, filename):
        path = "/".join((self.folder, "output", filename))
        self.review_model = Word2Vec.load_word2vec_format(path)

    def get_score_matrix(self):
        filename = "/".join((self.folder, "score"))
        if not os.path.exists(filename):
            prod_n = len(self.prod2idx)
            user_n = len(self.user2idx)

            self.matrix = lil_matrix((prod_n, user_n), dtype=np.float)

            with open(self.path, "r") as ins:
                for line in ins:
                    obj = json.loads(line)
                    # reviewText = obj["reviewText"]
                    # summary = obj["summary"]
                    reviewerID = obj["reviewerID"]
                    overall = obj["overall"]
                    asin = obj["asin"]
                    prod_idx = self.prod2idx[asin]
                    user_idx = self.user2idx[reviewerID]
                    self.matrix[prod_idx, user_idx] = overall
            np.save(filename, self.matrix)
        else:
            self.matrix = np.load(filename)

    def get_score_model(self):
        matrix = self.get_score_matrix()
        self.nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(matrix)
        return self.nbrs

    def get_neighers(self, prod_idx):
        distances, indices = self.nbrs.kneighbors(prod_idx)


exp = Exp("/home/sanqiang/Documents/data/Electronics_5.json", "amazon_electronics")
emb = exp.split()