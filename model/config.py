import os
import theano
class Config:
    def __init__(self):
        home = os.environ["HOME"]

        # for data
        self.path_data = "".join([home, "/data/yelp/review_processed_rest_interestword_DEC22.txt"])
        # self.path_data = "".join([home, "/data/yelp/sample.txt"])
        self.path_embed = "".join([home, "/data/glove/glove.processed.twitter.27B.200d.txt"])
        self.dim = 200
        self.neg_trials = 100

        # for model
        self.path_weight = "".join([home, "/model/chk/weight_", "<LOOP_IDX>"])
        if not os.path.exists(self.path_weight):
            os.makedirs(self.path_weight)
        self.path_checker = "".join([home, "/model/chk/checkpointweights.hdf5"])
        if not os.path.exists(self.path_checker):
            os.makedirs(self.path_checker)
        self.path_npy = "".join([home, "/model/npy/"])
        if not os.path.exists(self.path_npy):
            os.makedirs(self.path_npy)
        self.batch_size = 100000
        self.n_epoch = 100000

        # for framework
        theano.config.openmp = True
