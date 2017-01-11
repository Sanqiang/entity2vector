import os
import numpy as np
import random as rd
from scipy.sparse import *
from scipy.io import mmread, mmwrite
from config import *
import threading


home = os.environ["HOME"]

class DataProvider:
    def __init__(self, conf):
        self.conf = conf
        npy_checker = "".join([self.conf.path_npy, "idx2word.npy"])
        self.f_log = open(conf.path_logs, "w")
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
        self.tag2idx = {}
        self.idx2tag = []

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
                self.prod2idx[prod] = len(self.idx2prod)
                self.idx2prod.append(prod)

            tags = items[1]
            for tag in tags.split():
                if tag not in self.tag2idx:
                    self.tag2idx[tag] = len(self.idx2tag)
                    self.idx2tag.append(tag)

            words = items[2]
            for word in words.split():
                if word not in self.word2idx and word in self.temp_word_embedding:
                    self.word2idx[word] = len(self.idx2word)
                    self.idx2word.append(word)

        print("finish", "process idx")
        # process cor-occurrence data
        self.word_doc_cor_fmatrix = np.full(shape=(len(self.idx2word), len(self.idx2prod)), fill_value=False, dtype=np.bool)
        self.word_tag_cor_fmatrix = np.full(shape=(len(self.idx2word), len(self.idx2tag)), fill_value=False, dtype=np.bool)
        self.doc_tag_cor_fmatrix = np.full(shape=(len(self.idx2prod), len(self.idx2tag)), fill_value=False, dtype=np.bool)
        for line in open(self.conf.path_data, "r"):
            items = line.split("\t")

            if len(items) != 3:
                continue

            prod = items[0]
            prod_idx = self.prod2idx[prod]

            tags = items[1].split()
            words = items[2].split()

            for tag in tags:
                tag_idx = self.tag2idx[tag]
                self.doc_tag_cor_fmatrix[prod_idx, tag_idx] = True

                for word in words:
                    if word in self.temp_word_embedding:
                        word_idx = self.word2idx[word]
                        self.word_tag_cor_fmatrix[word_idx, tag_idx] = True

            for word in words:
                if word in self.temp_word_embedding:
                    word_idx = self.word2idx[word]
                    self.word_doc_cor_fmatrix[word_idx, prod_idx] = True
        print("finish", "cor-occurrence data")

        # process web embed
        self.word_embed = np.full(shape=(len(self.word2idx), self.conf.dim_word), fill_value=0, dtype=np.float64)
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
        np.save("".join([self.conf.path_npy, "word_embed"]), self.word_embed)
        np.save("".join([self.conf.path_npy, "idx2prod"]), self.idx2prod)
        np.save("".join([self.conf.path_npy, "idx2word"]), self.idx2word)
        np.save("".join([self.conf.path_npy, "idx2tag"]), self.idx2tag)
        np.save("".join([self.conf.path_npy, "word_doc_cor_fmatrix"]), self.word_doc_cor_fmatrix)
        np.save("".join([self.conf.path_npy, "word_tag_cor_fmatrix"]), self.word_tag_cor_fmatrix)
        np.save("".join([self.conf.path_npy, "doc_tag_cor_fmatrix"]), self.doc_tag_cor_fmatrix)
        # np.save(, self.word_doc_cor_smatrix)
        # mmwrite("".join([self.conf.path_npy, "word_doc_cor_smatrix"]), self.word_doc_cor_smatrix, field="integer")
        print("finish", "save")

    def load(self):
        self.word_embed = np.load("".join([self.conf.path_npy, "word_embed.npy"]))
        self.idx2prod = np.load("".join([self.conf.path_npy, "idx2prod.npy"]))
        self.idx2word = np.load("".join([self.conf.path_npy, "idx2word.npy"]))
        self.idx2tag = np.load("".join([self.conf.path_npy, "idx2tag.npy"]))
        self.word_doc_cor_fmatrix = np.load("".join([self.conf.path_npy, "word_doc_cor_fmatrix.npy"]))
        self.word_tag_cor_fmatrix = np.load("".join([self.conf.path_npy, "word_tag_cor_fmatrix.npy"]))
        self.doc_tag_cor_fmatrix = np.load("".join([self.conf.path_npy, "doc_tag_cor_fmatrix.npy"]))
        # self.word_doc_cor_smatrix = np.load("".join([self.conf.path_npy, "word_doc_cor_smatrix.npy"]))
        # self.word_doc_cor_smatrix = mmread("".join([self.conf.path_npy, "word_doc_cor_smatrix.mtx"])).todok()
        print("finish","load")

    def generate_init(self):
        self.word_doc_cor_smatrix = coo_matrix(self.word_doc_cor_fmatrix)
        self.word_tag_cor_smatrix = coo_matrix(self.word_tag_cor_fmatrix)
        self.doc_tag_cor_smatrix = coo_matrix(self.doc_tag_cor_fmatrix)


    def generate_data(self, batch_size):
        wd_word_idx = np.zeros((batch_size, 1))
        wd_pos_doc_idx = np.zeros((batch_size, 1))
        wd_neg_doc_idx = np.zeros((batch_size, 1))

        wt_word_idx = np.zeros((batch_size, 1))
        wt_pos_tag_idx = np.zeros((batch_size, 1))
        wt_neg_tag_idx = np.zeros((batch_size, 1))

        dt_pos_doc_idx = np.zeros((batch_size, 1))
        dt_pos_tag_idx = np.zeros((batch_size, 1))
        dt_neg_doc_idx = np.zeros((batch_size, 1))
        dt_neg_tag_idx = np.zeros((batch_size, 1))


        labels = np.zeros((batch_size, 1))
        append_data = True

        it_wd = (zip(self.word_doc_cor_smatrix.row, self.word_doc_cor_smatrix.col))
        it_wt = (zip(self.word_tag_cor_smatrix.row, self.word_tag_cor_smatrix.col))
        it_dt = (zip(self.doc_tag_cor_smatrix.row, self.doc_tag_cor_smatrix.col))
        while True:

            batch_idx = 0
            while batch_idx < batch_size:
                try:
                    word_idx, pos_doc_idx = next(it_wd)
                except StopIteration:
                    it_wd = zip(self.word_doc_cor_smatrix.row, self.word_doc_cor_smatrix.col)
                    word_idx, pos_doc_idx = next(it_wd)
                trials = 0
                while True:
                    neg_doc_idx = rd.randint(0, len(self.idx2prod) - 1)
                    trials += 1
                    if trials >= self.conf.neg_trials:
                        append_data = False
                        break
                    if not self.word_doc_cor_fmatrix[word_idx, neg_doc_idx]:
                        append_data = True
                        break
                if append_data:
                    wd_word_idx[batch_idx, 0] = word_idx
                    wd_pos_doc_idx[batch_idx, 0] = pos_doc_idx
                    wd_neg_doc_idx[batch_idx, 0] = neg_doc_idx
                    batch_idx += 1
            wd_data = {"wd_word_idx": wd_word_idx, "wd_pos_doc_idx":wd_pos_doc_idx, "wd_neg_doc_idx": wd_neg_doc_idx}

            batch_idx = 0
            while batch_idx < batch_size:
                try:
                    word_idx, pos_tag_idx = next(it_wt)
                except StopIteration:
                    it_wt = zip(self.word_tag_cor_smatrix.row, self.word_tag_cor_smatrix.col)
                    word_idx, pos_tag_idx = next(it_wt)
                trials = 0
                while True:
                    neg_tag_idx = rd.randint(0, len(self.idx2tag) - 1)
                    trials += 1
                    if trials >= self.conf.neg_trials:
                        append_data = False
                        break
                    if not self.word_tag_cor_fmatrix[word_idx, neg_tag_idx]:
                        append_data = True
                        break
                if append_data:
                    wt_word_idx[batch_idx, 0] = word_idx
                    wt_pos_tag_idx[batch_idx, 0] = pos_tag_idx
                    wt_neg_tag_idx[batch_idx, 0] = neg_tag_idx
                    batch_idx += 1
            wt_data = {"wt_word_idx": wt_word_idx, "wt_pos_tag_idx": wt_pos_tag_idx, "wt_neg_tag_idx": wt_neg_tag_idx}

            batch_idx = 0
            while batch_idx < batch_size:
                try:
                    pos_doc_idx, pos_tag_idx = next(it_dt)
                except StopIteration:
                    it_dt = zip(self.doc_tag_cor_smatrix.row, self.doc_tag_cor_smatrix.col)
                    pos_doc_idx, pos_tag_idx = next(it_dt)
                trials = 0
                while True:
                    neg_doc_idx = rd.randint(0, len(self.idx2prod) - 1)
                    trials += 1
                    if trials >= self.conf.neg_trials:
                        append_data = False
                        break
                    if not self.doc_tag_cor_fmatrix[neg_doc_idx, pos_tag_idx]:
                        append_data = True
                        break
                trials = 0
                while True:
                    neg_tag_idx = rd.randint(0, len(self.idx2tag) - 1)
                    trials += 1
                    if trials >= self.conf.neg_trials:
                        append_data = False
                        break
                    if not self.doc_tag_cor_fmatrix[pos_doc_idx, neg_tag_idx]:
                        append_data = True
                        break
                if append_data:
                    dt_pos_doc_idx[batch_idx, 0] = pos_doc_idx
                    dt_pos_tag_idx[batch_idx, 0] = pos_tag_idx
                    dt_neg_doc_idx[batch_idx, 0] = neg_doc_idx
                    dt_neg_tag_idx[batch_idx, 0] = neg_tag_idx
                    batch_idx += 1

            dt_data = {"dt_pos_doc_idx":dt_pos_doc_idx, "dt_pos_tag_idx": dt_pos_tag_idx, "dt_neg_doc_idx":dt_neg_doc_idx, "dt_neg_tag_idx":dt_neg_tag_idx}
            if not append_data:
                continue
            yield ({**wd_data, **wt_data, **dt_data},
                   {"final_merge": labels})

            wd_word_idx = np.zeros((batch_size, 1))
            wd_pos_doc_idx = np.zeros((batch_size, 1))
            wd_neg_doc_idx = np.zeros((batch_size, 1))

            wt_word_idx = np.zeros((batch_size, 1))
            wt_pos_tag_idx = np.zeros((batch_size, 1))
            wt_neg_tag_idx = np.zeros((batch_size, 1))

            dt_pos_doc_idx = np.zeros((batch_size, 1))
            dt_pos_tag_idx = np.zeros((batch_size, 1))
            dt_neg_doc_idx = np.zeros((batch_size, 1))
            dt_neg_tag_idx = np.zeros((batch_size, 1))



