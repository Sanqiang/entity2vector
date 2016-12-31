import os
import sys
import timeit
import numpy
from keras.models import Model
from keras.layers import Input
from keras.layers.core import *
from keras.layers.embeddings import *
from data import DataProvider
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
from config import Config

conf = Config()

# get data
dp = DataProvider(conf)
n_terms = len(dp.idx2word)
n_docs = len(dp.idx2prod)
dim = conf.dim
word_embed_data = np.array(dp.word_embed)
word_data = np.array(dp.word_data)
doc_pos_data = np.array(dp.doc_pos_data)
doc_neg_data = np.array(dp.doc_neg_data)
doc_embed_data = np.random.rand(len(dp.idx2prod), conf.dim)

print("finish data processing")

# define model
word_input = Input(shape=(1,), dtype ="int64", name ="word_idx")
doc_pos_input = Input(shape=(1,), dtype ="int64", name ="doc_pos_idx")
doc_neg_input = Input(shape=(1,), dtype ="int64", name ="doc_neg_idx")

word_embed = Embedding(output_dim=dim, input_dim=n_terms, input_length=1, name="word_embed",
                       weights=[word_embed_data], trainable=False)
doc_embed = Embedding(output_dim=dim, input_dim=n_docs, input_length=1, name="doc_embed",
                      weights=[doc_embed_data], trainable=True)

word_embed_ = word_embed(word_input)
doc_pos_embed_ = doc_embed(doc_pos_input)
doc_neg_embed_ = doc_embed(doc_neg_input)

# word_embed_ = Activation("sigmoid")(word_embed_)
# doc_pos_embed_ = Activation("softmax")(doc_pos_embed_)
# doc_neg_input = Activation("softmax")(doc_neg_input)

# word_embed_ = Reshape(input_shape=(1, dp.dim), target_shape=(dp.dim, 1))(word_embed_)
# doc_pos_embed_ = Reshape(input_shape=(1, dp.dim), target_shape=(dp.dim, 1))(doc_pos_embed_)
# doc_neg_embed_ = Reshape(input_shape=(1, dp.dim), target_shape=(dp.dim, 1))(doc_neg_embed_)

pos_layer = Merge(mode="dot", dot_axes=-1, name="pos_layer")([word_embed_, doc_pos_embed_])
neg_layer = Merge(mode="dot", dot_axes=-1, name="neg_layer")([word_embed_, doc_neg_embed_])
merge_layer = Merge(mode=lambda x: .5 - x[0] + x[1], output_shape=[None, 1], name="merge_layer")([pos_layer, neg_layer])

model = Model(input=[word_input, doc_pos_input, doc_neg_input], output=merge_layer)

def rawloss(x_train, x_test):
    return x_test * x_train

model.compile(optimizer='Adadelta', loss = {'merge_layer' : rawloss})

print("finish model compiling")
print(model.summary())

class my_checker_point(Callback):
    def __init__(self, model):
        self.loop_idx = 0
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        self.loop_idx += 1

    def on_epoch_end(self, epoch, logs={}):
        path = conf.path_weight.replace("<LOOP_IDX>", str(self.loop_idx))
        model.save_weights(path)


target = np.reshape(np.array([10] * len(word_data)), (len(word_data), 1, 1))
model.fit(
    {"word_idx":word_data, "doc_pos_idx":doc_pos_data, "doc_neg_idx":doc_neg_data},
    {"merge_layer":target},
    batch_size=conf.batch_size,nb_epoch=conf.n_epoch,validation_split = 0.1,
    callbacks=[my_checker_point(model), ModelCheckpoint(filepath=conf.path_checker, verbose = 1, save_best_only=True)])