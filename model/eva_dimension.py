from config import Config
from data import DataProvider
from gensim.models.word2vec import Word2Vec
import numpy as np
import os

flag = "Adam"
conf = Config(flag)

if not os.path.exists(conf.path_word_w2c):
    doc_embed = np.load(conf.path_doc_npy + ".npy")[0]
    dp = DataProvider(conf)

    # generate doc embedding file
    f = open(conf.path_doc_w2c,"w")
    f.write(str(len(dp.idx2prod)))
    f.write(" ")
    f.write(str(conf.dim_prod))
    f.write("\n")
    idx = 0
    batch = ""
    for word in dp.idx2prod:
        batch = "".join([batch, word])
        batch = "".join([batch, " "])

        for i in range(conf.dim_prod):
            batch = "".join([batch, str(doc_embed[idx][i])])
            batch = "".join([batch, " "])

        batch = "".join([batch, "\n"])
        idx += 1
        if len(batch) > 100000:
            f.write(batch)
            batch = ""
    f.write(batch)

if not os.path.exists(conf.path_word_w2c):

