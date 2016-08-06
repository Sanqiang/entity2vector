#word2vec pretraining imple
from stemmer import PorterStemmer
import json
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
import os.path
import re
from w2v_base import W2V_base

class W2V_cpp2(W2V_base):
    def __init__(self, path, folder, prod_sign=False, usr_sign=False, pos_sign=False):
        self.prod_sign = prod_sign
        self.usr_sign = usr_sign
        self.pos_sign = pos_sign
        W2V_base.__init__(self, path, folder)

    def process_vector(self):
        path_origin = "/home/sanqiang/data/glove/glove.twitter.27B.200d.txt"
        path_update = "/".join((self.folder, "wordvector.txt"))
        f_update = open(path_update, "w")
        f_origin = open(path_origin, "r")
        str_update = ""
        for line in f_origin:
            items = line.split(" ")
            word = items[0]
            if word not in self.word_count:
                continue
            nline = " ".join(items)
            str_update = "\n".join((str_update, nline))
        f_update.write(str_update)
        f_origin.close()
        f_update.close()

    def generate_word(self):
        f = open("/".join((self.folder, "pairword2.txt")), "w")
        for word, cnt in self.word_count:
            #cnt = self.word_count[word]
            f.write(word)
            f.write(" ")
            f.write(str(cnt))
            f.write("\n")

    def process(self):
        path_pair = "/".join((self.folder, "pairentity.txt"))
        f_pair = open(path_pair, "w")
        results = []
        n_pair = 0
        prod2idx = {}
        user2idx = {}
        for obj in self.data:
            prod = obj["prod"]
            user = obj["user"]
            text_data = obj["text_data"]

            for sent in text_data:
                for word, tag in sent:
                    if self.pos_sign and tag not in self.interest_tag:
                        continue
                    if self.prod_sign:
                        results.append([prod, word])
                        if prod not in prod2idx:
                            prod2idx[prod] = len(prod2idx)
                    if self.usr_sign:
                        results.append([user, word])
                        if user not in user2idx:
                            user2idx[user] = len(user2idx)

            if len(results) >= 10000:
                n_pair += len(results)
                print(len(results))
                for entity, word in results:
                    f_pair.write(str(entity))
                    f_pair.write(" ")
                    f_pair.write(str(word))
                    f_pair.write("\n")
                results = []

        n_pair += len(results)
        print(len(results))
        for entity, word in results:
            f_pair.write(str(entity))
            f_pair.write(" ")
            f_pair.write(str(word))
            f_pair.write("\n")

        #process prod
        path_prod = "/".join((self.folder, "prod.txt"))
        f_prod = open(path_prod, "w")
        for prod in prod2idx:
            f_prod.write(prod)
            f_prod.write("\n")

        #process user
        #process prod
        path_user = "/".join((self.folder, "prod.txt"))
        f_user = open(path_user, "w")
        for user in user2idx:
            f_prod.write(user)
            f_prod.write("\n")


        print("#prod", len(prod2idx))
        print("#user", len(user2idx))
        print("#pair", n_pair)


def main():
    w2v_cpp2 = W2V_cpp2("/home/sanqiang/data/yelp/review_rest.json", "yelp_rest_allalphaword_yelp_mincnt10_win10", prod_sign=True, pos_sign=True)
    print("init")
    w2v_cpp2.process()
    print("process")
    w2v_cpp2.process_vector()
    print("vector")
    w2v_cpp2.generate_word()

main()

