import numpy as np
import os
from gensim.models.word2vec import Word2Vec
import json

home = os.environ["HOME"]

class Vector:
    def load_word2vec_format(self, path):
        f = open(path, "r")
        line = f.readline()
        self.vocab_sizes = []
        vocab_size, self.vector_size = map(int, line.split())
        self.vocab_sizes.append(vocab_size)
        self.syn0 = np.zeros((vocab_size, self.vector_size), dtype=float)
        self.word2idx = {}
        self.idx2word = []
        self.cnt_word = 0
        self.init_business_info()

        def add_word(word, weights):
            if word not in self.word2idx:
                #process word2idx & idx2word
                self.word2idx[word] = self.cnt_word
                self.idx2word.append(word)
                #process syn0
                denom = 0
                for idx in range(self.vector_size):
                    denom += weights[idx]**2
                for idx in range(self.vector_size):
                    weights[idx] /= denom
                self.syn0[self.cnt_word] = weights
                #process cnt_word / idx
                self.cnt_word += 1

        for line_no, line in enumerate(f):
            parts = line.split()
            if len(parts) != self.vector_size+1:
                continue
                #raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
            word, weights = parts[0], list(map(float, parts[1:]))
            add_word(word, weights)

        return self.syn0

    def append_word2vec_format(self, path):
        f = open(path, "r")
        line = f.readline()
        vocab_size, self.vector_size = map(int, line.split())
        syn_temp = np.zeros((vocab_size, self.vector_size), dtype=float)
        self.syn0 = np.concatenate((self.syn0, syn_temp))
        self.vocab_sizes.append(vocab_size)

        def add_word(word, weights):
            if word not in self.word2idx:
                #process word2idx & idx2word
                self.word2idx[word] = self.cnt_word
                self.idx2word.append(word)
                #process syn0
                denom = 0
                for idx in range(self.vector_size):
                    denom += weights[idx] ** 2
                for idx in range(self.vector_size):
                    weights[idx] /= denom
                self.syn0[self.cnt_word] = weights
                #process cnt_word / idx
                self.cnt_word += 1

        for line_no, line in enumerate(f):
            parts = line.split()
            if len(parts) != self.vector_size+1:
                raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
            word, weights = parts[0], list(map(float, parts[1:]))
            add_word(word, weights)


    def most_similar(self, query_prod_id, restrict_inds = None, k = 10):
        if restrict_inds is None:
            dists = np.dot(self.syn0[query_prod_id,], self.syn0.T)
        else:
            dists = np.dot(self.syn0[query_prod_id,], self.syn0[restrict_inds,].T)
        best_idxs = np.argsort(dists)[::-1][:k]
        best = [restrict_inds[best_idx] for best_idx in best_idxs]
        return zip(best,dists[best_idxs])

    def init_business_info(self):
        f = open("".join((home, "/data/yelp/", "business.json")), "r")
        self.business_info = {}
        for line in f:
            obj = json.loads(line)
            business_id = obj["business_id"]
            name = obj["name"].replace("\n", "")
            full_address = obj["full_address"].replace("\n", "")
            self.business_info[business_id] = "\t".join([name, full_address])

if __name__ == '__main__':
    if False:
        vec = Vector()
        vec.most_similar_ori()
    if True:
        name = "mode_flag_2_2012.000000"
        path_prod_vector = "".join((home, "/data/model/", name, ".prod.vec"))
        path_tag_vector = "".join((home, "/data/model/", name, ".tag.vec"))
        path_word_vector = "".join((home, "/data/model/", name, ".word.vec"))

        check_rest = True

        vec = Vector()
        vec.load_word2vec_format(path_word_vector)
        vec.append_word2vec_format(path_prod_vector)
        vec.append_word2vec_format(path_tag_vector)
        print("init")
        while True:
            word = input()
            if check_rest:
                pairs = vec.most_similar(vec.word2idx[word], restrict_inds=range(0, vec.vocab_sizes[0] + vec.vocab_sizes[1] + vec.vocab_sizes[2]))
            else:
                pairs = vec.most_similar(vec.word2idx[word])
            for prod_id, dist in pairs:
                if check_rest and vec.idx2word[prod_id] in vec.business_info:
                    line = "\t".join([vec.business_info[vec.idx2word[prod_id]],vec.idx2word[prod_id].replace("\n", ""), str(dist).replace("\n", "")])
                else:
                    line = "\t".join([vec.idx2word[prod_id].replace("\n",""), str(dist).replace("\n","")])
                print(line)