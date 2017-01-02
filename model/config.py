import os
import theano
class Config:
    def __init__(self):
        home = os.environ["HOME"]

        # for data
        self.path_data = "".join([home, "/data/yelp/review_processed_rest_interestword_DEC22.txt"])
        # self.path_data = "".join([home, "/data/yelp/sample.txt"])
        self.path_embed = "".join([home, "/data/glove/glove.processed.twitter.27B.200d.txt"])

        self.dim_word = 200
        self.dim_prod = 150

        self.neg_trials = 100

        # for model
        self.path_weight = "".join([home, "/data/model/chk/weight"])
        self.path_checker = "".join([home, "/data/model/chk/checkpointweights.hdf5"])
        self.path_npy = "".join([home, "/data/model/npy/"])
        self.batch_size = 100000
        self.n_epoch = 100000

        # for framework
        # theano.config.openmp = True

        # for save
        self.path_doc_npy = "".join([home, "/data/model/chk/doc"])
        self.path_word_npy = "".join([home, "/data/model/chk/word"])
        self.path_doc_w2c = "".join([home, "/data/model/chk/doc.txt"])
        self.path_word_w2c = "".join([home, "/data/model/chk/word.txt"])
