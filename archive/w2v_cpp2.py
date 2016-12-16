#word2vec pretraining imple
from stemmer import PorterStemmer
import json
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
import os.path
import re
import time
import datetime
from w2v_base import W2V_base
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import dok_matrix
from gensim.models import Word2Vec
import numpy as np
import math
import os
home = os.environ["HOME"]

class W2V_cpp2(W2V_base):
    def __init__(self, path, folder, prod_sign=False, usr_sign=False, pos_sign=False):
        self.method = "LDA"
        self.prod_sign = prod_sign
        self.usr_sign = usr_sign
        self.pos_sign = pos_sign
        W2V_base.__init__(self, path, folder)
        self.k = 5
        self.train_path = "/".join((self.folder, "train.json"))
        self.test_path = "/".join((self.folder, "test.json"))

        #process dict
        for prod_id in self.idx2prod.keys():
            prod = self.idx2prod[prod_id]
            n_prod_id = prod_id - len(self.word_count) - 1
            del self.idx2prod[prod_id]
            self.idx2prod[n_prod_id] = prod
            self.prod2idx[prod] = n_prod_id

        for user_id in self.idx2user.keys():
            user = self.idx2user[user_id]
            n_user_id = user_id - len(self.word_count) - len(self.prod2idx) - 1
            del self.idx2user[user_id]
            self.idx2user[n_user_id] = user
            self.user2idx[user] = n_user_id



    def process_vector(self):
        #path_origin = "/home/sanqiang/data/glove/glove.twitter.27B.200d.txt"
        idx2interested_words = {} #idx follow the word vector pretraining file
        interested_words2idx = {}

        path_origin = "/Users/zhaosanqiang916/data/glove/glove.twitter.27B.200d.txt"
        path_update = "/".join((self.folder, "wordvector.txt"))
        f_update = open(path_update, "w")
        f_origin = open(path_origin, "r")
        str_update = ""
        words = [ala[0] for ala in self.word_count]
        for line in f_origin:
            items = line.split(" ")
            word = items[0]
            if word not in words: #only consider the word occur in the word vector and our data set
                continue
            nline = " ".join(items)
            str_update = "".join((str_update, nline))

            idx2interested_words[len(idx2interested_words)] = word
            interested_words2idx[word] = len(interested_words2idx)

            if len(str_update) > 10000:
                f_update.write(str_update)
                str_update = ""

        f_update.write(str_update)
        f_origin.close()
        f_update.close()

        return idx2interested_words,interested_words2idx

    def generate_word(self, interested_words2idx):
        f = open("/".join((self.folder, "pairword2.txt")), "w")
        for word,cnt in self.word_count:
            if word not in interested_words2idx:
                continue
            f.write(word)
            f.write(" ")
            f.write(str(cnt))
            f.write("\n")

    def process(self, interested_words2idx):
        path_pair = "/".join((self.folder, "pairentity.txt"))
        path_pair_concrete = "/".join((self.folder, "pairentity_concrete.txt"))
        f_pair = open(path_pair, "w")
        f_pair_concrete = open(path_pair_concrete, "w")
        results = []
        results_concrete = []
        n_pair = 0
        prod2idx = OrderedDict() #only for current dataset not self ones
        user2idx = OrderedDict()

        #populate prod2idx and user2idx
        for obj in self.data:
            prod = obj["prod"]
            user = obj["user"]
            text_data = obj["text_data"]

            for sent in text_data:
                for word, tag in sent:
                    if word == -1 or self.idx2word[word] not in interested_words2idx:
                        continue
                    if self.pos_sign and tag not in self.interest_tag:
                        continue
                    if self.prod_sign:
                        if prod not in prod2idx:
                            prod2idx[prod] = len(prod2idx)
                    if self.usr_sign:
                        if user not in user2idx:
                            user2idx[user] = len(user2idx)

        for obj in self.data:
            prod = obj["prod"]
            user = obj["user"]
            text_data = obj["text_data"]

            for sent in text_data:
                for word, tag in sent:
                    if word == -1 or self.idx2word[word] not in interested_words2idx:
                        continue
                    word = interested_words2idx[self.idx2word[word]]
                    word_concrete = self.idx2word[word]
                    if self.pos_sign and tag not in self.interest_tag:
                        continue
                    if self.prod_sign:
                        results.append([prod2idx[prod], word])
                        results_concrete.append([prod2idx[prod], word_concrete])
                    if self.usr_sign:
                        results.append([user2idx[user], word])
                        results_concrete.append([user2idx[user], word_concrete])
            if len(results) >= 10000:
                n_pair += len(results)
                print(len(results))
                for entity, word in results:
                    f_pair.write(str(entity))
                    f_pair.write(" ")
                    f_pair.write(str(word))
                    f_pair.write("\n")
                results = []
                for entity, word in results_concrete:
                    f_pair_concrete.write(str(entity))
                    f_pair_concrete.write(" ")
                    f_pair_concrete.write(str(word))
                    f_pair_concrete.write("\n")
                results_concrete = []

        n_pair += len(results)
        print(len(results))
        for entity, word in results:
            f_pair.write(str(entity))
            f_pair.write(" ")
            f_pair.write(str(word))
            f_pair.write("\n")
        for entity, word in results_concrete:
            f_pair_concrete.write(str(entity))
            f_pair_concrete.write(" ")
            f_pair_concrete.write(str(word))
            f_pair_concrete.write("\n")
        results_concrete = []
        #process prod
        path_prod = "/".join((self.folder, "prod.txt"))
        f_prod = open(path_prod, "w")
        for prod in prod2idx:
            f_prod.write(prod)
            f_prod.write("_")
            f_prod.write(str(prod2idx[prod]))
            f_prod.write("\n")

        #process user
        #process prod
        path_user = "/".join((self.folder, "user.txt"))
        f_user = open(path_user, "w")
        for user in user2idx:
            f_user.write(user)
            f_user.write("_")
            f_user.write(str(user2idx[user]))
            f_user.write("\n")


        print("#prod", len(prod2idx))
        print("#user", len(user2idx))
        print("#pair", n_pair)

    #exp
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
                date_str = obj["date"]
                date_time = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                time_time = time.mktime(date_time.timetuple())
                heapq.heappush(h, time_time)
        split = h[int(len(h) * 0.8)]

        f_train = open(self.train_path, "w")
        f_test = open(self.test_path, "w")
        with open(self.path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                date_str = obj["date"]
                date_time = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                time_time = time.mktime(date_time.timetuple())
                if time_time < split: # go to train
                    f_train.write(line)
                else:
                    f_test.write(line)
        return split

    def populate_score(self, path):
        matrix = dok_matrix((len(self.user2idx), len(self.prod2idx)), dtype=np.float)
        with open(path, "r", encoding=self.file_encoding) as ins:
            for line in ins:
                title, text, user, prod, rating = self.line_parser(line)
                user_idx = self.user2idx[user]
                prod_idx = self.prod2idx[prod]
                if user_idx >= len(self.user2idx) or prod_idx >= len(self.prod2idx):
                    print("out range")
                matrix[user_idx, prod_idx] = rating
        return matrix

    def populate_entity(self, path_vec,path_entity, prod_model = True):
        self.path_vec = path_vec
        self.path_entity = path_entity
        self.prod_model = prod_model
        self.entity_model = Word2Vec.load_word2vec_format(path_vec)

        self.entity2idx = {}
        self.idx2entity = OrderedDict()

        f = open(path_entity, "r")
        for line in f:
            if self.method == "LDA":
                entity = line[0:-1]
                idx = len(self.entity2idx)

                if entity not in self.entity2idx:
                    self.entity2idx[entity] = idx
                    self.idx2entity[idx] = entity
                else:
                    print("dup?")
            else:
                entity = line[0:line.rindex("_")]
                idx = int(line[1 + line.rindex("_"):])

                self.entity2idx[entity] = idx
                self.idx2entity[idx] = entity



    def predict(self, product_idx, user_idx):
        if self.prod_model:
            #find similar prod (which user already rate)
            inds = (self.matrix[user_idx,:] > 0).indices
            inds =[self.entity2idx[self.idx2prod[ind]] for ind in inds if self.idx2prod[ind] in self.entity2idx]
            entity_idx = self.entity2idx[self.idx2prod[product_idx]]
            entity = self.idx2entity[entity_idx]
            if self.method == "LDA":
                list = self.entity_model.most_similar(entity, topn=self.k,
                                                      restrict_vocab=inds)
            else:
                list = self.entity_model.most_similar("_".join((entity, str(entity_idx))), topn=self.k, restrict_vocab=inds)

            if len(list) == 0:
                if len(inds) == 0:
                    return 0
                else:
                    print("x")
                    return 0


            preidct_score = 0
            denom = 0
            for pair in list:
                pair_ent = pair[0]
                #prod = pair_ent[0:pair_ent.rindex("_")]
                if self.method == "LDA":
                    prod_id = self.prod2idx[pair_ent]
                else:
                    prod_id = inds[int(pair_ent[1+pair_ent.rindex("_"):])]
                dist = pair[1]
                rating = self.matrix[user_idx, prod_id]
                #preidct_score += rating * dist
                #denom += dist
                preidct_score += rating
                denom += 1
            preidct_score /= denom
            return preidct_score
        else:
            # find similar user (which prod already rate)
            inds = (self.matrix.T[product_idx, :] > 0).indices
            inds = [self.entity2idx[self.idx2user[ind]] for ind in inds if self.idx2user[ind] in self.entity2idx]
            if user_idx not in self.idx2user or self.idx2user[user_idx] not in self.entity2idx:
                print(self.idx2user[user_idx], "\n")
                return 0
            entity_idx = self.entity2idx[self.idx2user[user_idx]]
            entity = self.idx2entity[entity_idx]
            list = self.entity_model.most_similar("_".join((entity, str(entity_idx))), topn=self.k, restrict_vocab=inds)

            if len(list) == 0:
                if len(inds) == 0:
                    return 0
                else:
                    print("x")
                    return 0

            preidct_score = 0
            denom = 0
            for pair in list:
                pair_ent = pair[0]
                # prod = pair_ent[0:pair_ent.rindex("_")]
                usr_id = inds[int(pair_ent[1 + pair_ent.rindex("_"):])]
                dist = pair[1]
                rating = self.matrix[usr_id, product_idx]
                # preidct_score += rating * dist
                # denom += dist
                preidct_score += rating
                denom += 1
            preidct_score /= denom
            return preidct_score


    def test(self):
        rmse = []
        self.matrix = self.populate_score(self.train_path)
        self.tmatrix = self.populate_score(self.test_path)
        for user_idx, prod_idx in self.tmatrix.keys():
            truth_score = self.tmatrix[user_idx, prod_idx]
            preidct_score = self.predict(prod_idx, user_idx)
            if preidct_score is None or preidct_score == 0 or math.isnan(preidct_score):
                continue
            rmse.append((truth_score - preidct_score)**2)
        print(np.mean(rmse))


def main():
    #w2v_cpp2 = W2V_cpp2("/home/sanqiang/data/yelp/review_rest.json", "yelp_rest_allalphaword_yelp_mincnt10_win10", prod_sign=True, pos_sign=True)
    w2v_cpp2 = W2V_cpp2("".join([home, "/data/yelp/review_rest.json"]), "yelp_rest_prod",
                        pos_sign=True, usr_sign=False, prod_sign=True)
    print("init")

    if False:
        print("vector")
        idx2interested_words, interested_words2idx = w2v_cpp2.process_vector()
        w2v_cpp2.generate_word(interested_words2idx)

        print("process")
        w2v_cpp2.process(interested_words2idx)

    if True:
        #w2v_cpp2.split()

        #w2v_cpp2.populate_entity("/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod/output/model_75", "/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod/prod.txt", prod_model=True)
        #w2v_cpp2.populate_entity("/Users/zhaosanqiang916/git/entity2vector/yelp_rest_user_aword/output/model_75","/Users/zhaosanqiang916/git/entity2vector/yelp_rest_user_aword/user.txt",prod_model=F)
        w2v_cpp2.populate_entity("/Users/zhaosanqiang916/git/entity2vector/lda/model.txt",
                                  "/Users/zhaosanqiang916/git/entity2vector/lda/prod.txt", prod_model=True)

        print("LDA", w2v_cpp2.k, w2v_cpp2.prod_model, w2v_cpp2.path_vec, w2v_cpp2.path_entity)
        w2v_cpp2.test()



main()

