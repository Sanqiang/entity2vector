import os
import numpy as np
import random
home = os.environ["HOME"]

class DataProvider:
    def __init__(self, conf):
        self.conf = conf
        npy_checker = "".join([self.conf.path_npy, "word_data.npy"])
        if os.path.exists(npy_checker):
            print("find npy file", npy_checker)
            self.load()
        else:
            print("not find npy file", npy_checker)
            self.process()

    def process(self):
        self.word2idx = {}
        self.idx2word = []
        self.prod2idx = {}
        self.idx2prod = []

        self.word_data = []
        self.doc_pos_data = []
        self.doc_neg_data = []

        self.process_word_embed()
        self.process_data()

        self.save()

    def process_data(self):
        # process idx
        for line in open(self.conf.path_data, "r"):
            items = line.split("\t")
            if len(items) != 3:
                continue

            prod = items[0]
            if prod not in self.prod2idx:
                self.prod2idx[prod] = len(self.prod2idx)
                self.idx2prod.append(prod)

            words = items[2]
            for word in words.split():
                if word not in self.word2idx and word in self.temp_word_embedding:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)

        print("finish", "process idx")
        # process cor-occurrence data
        self.checker = np.full(shape=(len(self.word2idx), len(self.prod2idx)), fill_value=False, dtype=np.bool)
        for line in open(self.conf.path_data, "r"):
            items = line.split("\t")

            if len(items) != 3:
                continue

            prod = items[0]
            prod_idx = self.prod2idx[prod]
            words = items[2]
            for word in words.split():
                if word in self.temp_word_embedding:
                    word_idx = self.word2idx[word]
                    self.checker[word_idx, prod_idx] = True
        print("finish", "cor-occurrence data")
        # process data
        for line in open(self.conf.path_data, "r"):
            items = line.split("\t")
            if len(items) != 3:
                continue

            prod = items[0]
            prod_idx = self.prod2idx[prod]
            words = items[2]
            for word in words.split():
                if word in self.temp_word_embedding:
                    word_idx = self.word2idx[word]
                    self.word_data.append(word_idx)
                    self.doc_pos_data.append(prod_idx)

                    trials = 0
                    while True:
                        neg_prod_idx = random.randint(0, len(self.prod2idx) - 1)
                        trials += 1
                        if not self.checker[word_idx, neg_prod_idx] or trials >= self.conf.neg_trials:
                            break
                    self.doc_neg_data.append(neg_prod_idx)
        print("finish", "data")
        # process web embed
        self.word_embed = np.full(shape=(len(self.word2idx), self.conf.dim), fill_value=0, dtype=np.float64)
        for word in self.idx2word:
            word_idx = self.word2idx[word]
            self.word_embed[word_idx,] = self.temp_word_embedding[word]
        print("finish", "web embed")

    def process_word_embed(self):
        self.temp_word_embedding = {}
        for line in open(self.conf.path_embed, "r"):
            items = line.split()
            word = items[0]
            self.temp_word_embedding[word] = [float(val) for val in items[1:]]
        print("finish", "temp_word_embedding")

    def save(self):
        np.save("".join([self.conf.path_npy, "word_data"]), np.array(self.word_data))
        np.save("".join([self.conf.path_npy, "doc_pos_data"]), np.array(self.doc_pos_data))
        np.save("".join([self.conf.path_npy, "doc_neg_data"]), np.array(self.doc_neg_data))
        np.save("".join([self.conf.path_npy, "word_embed"]), self.word_embed)
        np.save("".join([self.conf.path_npy, "idx2prod"]), self.idx2prod)
        np.save("".join([self.conf.path_npy, "idx2word"]), self.idx2word)
        np.save("".join([self.conf.path_npy, "prod2idx"]), self.prod2idx)
        np.save("".join([self.conf.path_npy, "word2idx"]), self.word2idx)
        print("finish", "save")

    def load(self):
        self.word_data = np.load("".join([self.conf.path_npy, "word_data.npy"]))
        self.doc_pos_data = np.load("".join([self.conf.path_npy, "doc_pos_data.npy"]))
        self.doc_neg_data = np.load("".join([self.conf.path_npy, "doc_neg_data.npy"]))
        self.word_embed = np.load("".join([self.conf.path_npy, "word_embed.npy"]))
        self.idx2prod = np.load("".join([self.conf.path_npy, "idx2prod.npy"]))
        self.idx2word = np.load("".join([self.conf.path_npy, "idx2word.npy"]))
        self.prod2idx = np.load("".join([self.conf.path_npy, "prod2idx.npy"]))
        self.word2idx = np.load("".join([self.conf.path_npy, "word2idx.npy"]))
        print("finish","load")




