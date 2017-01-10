from config import Config
from data import DataProvider
from gensim.models.word2vec import Word2Vec
import numpy as np
import os
from scipy.stats import logistic

flag = "Adam"
conf = Config(flag)

model = np.load(conf.path_model_npy + ".npy")
word_embed = model[0]
prod_embed = model[1]
transfer_w = model[2]
transfer_b = model[3]

dp = DataProvider(conf)

weight = np.dot(word_embed, transfer_w)
weight = logistic.cdf(np.add(weight, transfer_b))

for topic_id in range(conf.dim_item):
    word_ids = weight[:,topic_id]
    word_ids = np.argsort(word_ids)[::-1][:50]
    words = [(dp.idx2word[word_id], weight[word_id, topic_id]) for word_id in word_ids if weight[word_id, topic_id] > .9]
    print("Topic", topic_id)
    print(words)
    print("=========================\n")

print("finish")







