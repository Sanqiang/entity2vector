import numpy as np

class vector:
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

        def add_word(word, weights):
            if word not in self.word2idx:
                #process word2idx & idx2word
                self.word2idx[word] = self.cnt_word
                self.idx2word.append(word)
                #process syn0
                self.syn0[self.cnt_word] = weights
                #process cnt_word / idx
                self.cnt_word += 1

        for line_no, line in enumerate(f):
            parts = line.split()
            if len(parts) != self.vector_size+1:
                raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
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
                self.syn0[self.cnt_word] = weights
                #process cnt_word / idx
                self.cnt_word += 1

        for line_no, line in enumerate(f):
            parts = line.split()
            if len(parts) != self.vector_size+1:
                raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
            word, weights = parts[0], list(map(float, parts[1:]))
            add_word(word, weights)


    def most_similar(self, query, k = 10):
        dists = np.dot(query, self.syn0)
        best = np.argmax(dists)
        return best[k]