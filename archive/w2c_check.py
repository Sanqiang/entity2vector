from sklearn.neighbors import NearestNeighbors
import numpy as np
from gensim.models.word2vec import Word2Vec
import os
home = os.environ["HOME"]

if False:
    #path = "/home/sanqiang/git/entity2vector/yelp_nv/result/multi_thread_446"
    path = "/home/sanqiang/data/glove/glove.twitter.27B.200d.txt"
    f = open(path, "r")
    word2idx = {}
    idx2word = []
    idx2vector = []
    idx = 0
    for line in f:
        items = line.split(" ")
        word = items[0]
        vector = [float(item) for item in items[1:-1]]
        if(len(vector)) < 100:
            print(line)
            break
        word2idx[word] = idx
        idx2word.append(word)
        idx2vector.append(vector)
        idx+=1


    idx2vector = np.array(idx2vector)

    k = 15
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', metric="euclidean").fit(idx2vector)

    while True:
        word = input()
        distances, indices = nbrs.kneighbors(idx2vector[word2idx[word]].reshape(1, -1))
        for i in range(k):
            nword = idx2word[indices[0][i]]
            print(nword)
        print("=======")
if True:
    #path = "".join([home, "/git/entity2vector/archive/yelp_rest_prod/output_noneg/model_22"])
    path = "".join([home, "/data/model/all_on5.prod.vec"])
    model = Word2Vec.load_word2vec_format(path)
    print("finished init")
    while True:
        word = input()
        words = model.most_similar(word)
        print(words)
        print("=======")

