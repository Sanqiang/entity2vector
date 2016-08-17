#word2vec pretraining imple
from stemmer import PorterStemmer
import json
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
import os.path
import re
from w2v_base import W2V_base
from collections import OrderedDict

class W2V_cpp2(W2V_base):
    def __init__(self, path, folder, prod_sign=False, usr_sign=False, pos_sign=False):
        self.prod_sign = prod_sign
        self.usr_sign = usr_sign
        self.pos_sign = pos_sign
        W2V_base.__init__(self, path, folder)

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


def main():
    #w2v_cpp2 = W2V_cpp2("/home/sanqiang/data/yelp/review_rest.json", "yelp_rest_allalphaword_yelp_mincnt10_win10", prod_sign=True, pos_sign=True)
    w2v_cpp2 = W2V_cpp2("/Users/zhaosanqiang916/data/yelp/review_rest.json", "yelp_rest_prod",
                        pos_sign=True, usr_sign=False, prod_sign=True)
    print("init")

    print("vector")
    idx2interested_words, interested_words2idx = w2v_cpp2.process_vector()
    w2v_cpp2.generate_word(interested_words2idx)

    print("process")
    w2v_cpp2.process(interested_words2idx)




main()

