# plain entity 2 vec model
import os
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
dim = conf.dim_word
word_embed_data = np.array(dp.word_embed)
word_data = np.array(dp.word_data)
doc_pos_data = np.array(dp.doc_pos_data)
doc_neg_data = np.array(dp.doc_neg_data)
doc_embed_data = np.random.rand(len(dp.idx2prod), conf.dim_word)

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

word_embed_ = Flatten()(word_embed_)
doc_pos_embed_ = Flatten()(doc_pos_embed_)
doc_neg_embed_ = Flatten()(doc_neg_embed_)

pos_layer = Merge(mode="dot", dot_axes=-1, name="pos_layer")([word_embed_, doc_pos_embed_])
neg_layer = Merge(mode="dot", dot_axes=-1, name="neg_layer")([word_embed_, doc_neg_embed_])
merge_layer = Merge(mode=lambda x: .5 - x[0] + x[1], output_shape=[1], name="merge_layer")([pos_layer, neg_layer])

model = Model(input=[word_input, doc_pos_input, doc_neg_input], output=merge_layer)

def rawloss(x_train, x_test):
    return x_test * x_train

model.compile(optimizer='Adadelta', loss = {'merge_layer' : rawloss})

print("finish model compiling")
print(model.summary())

class my_checker_point(Callback):
    def __init__(self, doc_embed, word_embed):
        self.loop_idx = 0
        self.doc_embed = doc_embed
        self.word_embed = word_embed

    def on_epoch_end(self, epoch, logs={}):
        np.save(conf.path_doc_npy, self.doc_embed.get_weights())
        np.save(conf.path_word_npy, self.word_embed.get_weights())



target = np.array([1] * len(word_data))
if os.path.exists(conf.path_checker):
    model.load_weights(conf.path_checker)
model.fit(
    {"word_idx":word_data, "doc_pos_idx":doc_pos_data, "doc_neg_idx":doc_neg_data},
    {"merge_layer":target},
    batch_size=conf.batch_size,nb_epoch=conf.n_epoch,validation_split = 0.1,
    callbacks=[my_checker_point(doc_embed, word_embed), ModelCheckpoint(filepath=conf.path_checker, verbose = 1, save_best_only=True)])