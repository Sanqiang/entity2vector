import os
import theano
class Config:
    def __init__(self, flag):
        home = os.environ["HOME"]
        self.flag = flag
        # for data
        self.path_data = "".join([home, "/data/yelp/review_processed_rest_interestword_DEC22.txt"])
        # self.path_data = "".join([home, "/data/yelp/sample.txt"])
        self.path_embed = "".join([home, "/data/glove/glove.processed.twitter.27B.200d.txt"])

        self.dim_word = 200
        self.dim_prod = 200

        self.neg_trials = 100

        # for model
        self.path_weight = "".join([home, "/data/model/chk_",self.flag , "/weight"])
        if not os.path.exists(os.path.dirname(self.path_weight)):
            os.mkdir(os.path.dirname(self.path_weight))
        self.path_checker = "".join([home, "/data/model/chk_",self.flag, "/checkpointweights.hdf5"])
        if not os.path.exists(os.path.dirname(self.path_checker)):
            os.mkdir(os.path.dirname(self.path_checker))
        self.path_npy = "".join([home, "/data/model/npy/"])
        if not os.path.exists(os.path.dirname(self.path_npy)):
            os.mkdir(os.path.dirname(self.path_npy))
        self.batch_size = 100000
        self.n_epoch = 100000
        # self.sample_per_epoch = 19200000
        self.sample_per_epoch = 1920

        # for framework
        theano.config.openmp = False

        # for save
        self.path_doc_npy = "".join([home, "/data/model/chk_",self.flag,"/doc"])
        self.path_word_npy = "".join([home, "/data/model/chk_",self.flag,"/word"])
        self.path_model_npy = "".join([home, "/data/model/chk_",self.flag,"/model"])
        self.path_doc_w2c = "".join([home, "/data/model/chk_",self.flag,"/doc.txt"])
        self.path_word_w2c = "".join([home, "/data/model/chk_",self.flag,"/word.txt"])
        if not os.path.exists(os.path.dirname(self.path_doc_npy)):
            os.mkdir(os.path.dirname(self.path_doc_npy))
        if not os.path.exists(os.path.dirname(self.path_word_npy)):
            os.mkdir(os.path.dirname(self.path_word_npy))
        if not os.path.exists(os.path.dirname(self.path_model_npy)):
            os.mkdir(os.path.dirname(self.path_model_npy))
        if not os.path.exists(os.path.dirname(self.path_doc_w2c)):
            os.mkdir(os.path.dirname(self.path_doc_w2c))
        if not os.path.exists(os.path.dirname(self.path_word_w2c)):
            os.mkdir(os.path.dirname(self.path_word_w2c))

        self.path_logs = "".join([home, "/data/model/log/", self.flag, ".log"])
        if not os.path.exists(os.path.dirname(self.path_logs)):
            os.mkdir(os.path.dirname(self.path_logs))

