from keras.callbacks import Callback
import numpy as np
import theano.tensor as T

class my_checker_point(Callback):
    def __init__(self, doc_embed, word_embed, model, conf):
        self.conf = conf
        self.loop_idx = 0
        self.doc_embed = doc_embed
        self.word_embed = word_embed
        self.model = model

    def on_epoch_end(self, epoch, logs={}):
        np.save(self.conf.path_doc_npy, self.doc_embed.get_weights())
        np.save(self.conf.path_word_npy, self.word_embed.get_weights())
        np.save(self.conf.path_model_npy, self.model.get_weights())

class my_value_checker(Callback):
    def __init__(self, models):
        self.models = models

    def on_batch_end(self, epoch, logs={}):
        for model_loop in self.models:
            temp = T.cast(model_loop, "float32")
            print(temp.eval())