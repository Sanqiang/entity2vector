# plain entity 2 vec model
import os
from keras.models import Model
from keras.layers import Input
from keras.layers.core import *
from keras.layers.embeddings import *
from model.layers import *
from data import DataProvider
from keras.callbacks import ModelCheckpoint
from model.callbacks import *
import numpy as np
from config import Config
from keras.optimizers import *
import numpy as np
import theano

test = True
flag = "naive_largebatch_Adam"
conf = Config(flag)
print(flag)

# get data
dp = DataProvider(conf)
n_terms = len(dp.idx2word)
n_docs = len(dp.idx2prod)
word_embed_data = np.array(dp.word_embed)
word_data = np.array(dp.word_data)
doc_pos_data = np.array(dp.doc_pos_data)
doc_neg_data = np.array(dp.doc_neg_data)

doc_embed_data = np.random.rand(len(dp.idx2prod), conf.dim_prod)
word_transfer_W = 100 * np.random.rand(conf.dim_word, conf.dim_prod)
word_transfer_b = np.random.rand(conf.dim_prod)
print("finish data processing")

# define model
word_input = Input(shape=(1,), dtype ="int64", name ="word_idx")
doc_pos_input = Input(shape=(1,), dtype ="int64", name ="doc_pos_idx")
doc_neg_input = Input(shape=(1,), dtype ="int64", name ="doc_neg_idx")

word_embed = Embedding(output_dim=conf.dim_word, input_dim=n_terms, input_length=1, name="word_embed",
                       weights=[word_embed_data], trainable=False)
doc_embed = Embedding(output_dim=conf.dim_prod, input_dim=n_docs, input_length=1, name="doc_embed",
                      weights=[doc_embed_data], trainable=True)

word_embed_ = word_embed(word_input)
doc_pos_embed_ = doc_embed(doc_pos_input)
doc_neg_embed_ = doc_embed(doc_neg_input)

word_flatten = Flatten()
word_embed_ = word_flatten(word_embed_)
word_embed_ = Dense(activation="sigmoid", output_dim=conf.dim_prod, input_dim=conf.dim_word, trainable=True,
                    weights=[word_transfer_W, word_transfer_b], name="word_transfer")(word_embed_)

doc_pos_flatten = Flatten()
doc_neg_flatten = Flatten()
doc_pos_embed_ = doc_pos_flatten(doc_pos_embed_)
doc_neg_embed_ = doc_neg_flatten(doc_neg_embed_)
doc_pos_embed_ = Activation(activation="softmax", name="doc_pos_act")(doc_pos_embed_)
doc_neg_embed_ = Activation(activation="softmax", name="doc_neg_act")(doc_neg_embed_)

pos_layer = Merge(mode="dot", dot_axes=-1, name="pos_layer")
pos_layer_ = pos_layer([word_embed_, doc_pos_embed_])
neg_layer = Merge(mode="dot", dot_axes=-1, name="neg_layer")
neg_layer_ = neg_layer([word_embed_, doc_neg_embed_])
merge_layer = Merge(mode="concat", concat_axis=-1, name="merge_layer")
merge_layer_ = merge_layer([pos_layer_, neg_layer_])

# move the margin loss into loss function rather than merge layer
# merge_layer = Merge(mode=lambda x: 0.5 - x[0] + x[1], output_shape=[1], name="merge_layer")
# merge_layer_ = merge_layer([pos_layer_, neg_layer_])

model = Model(input=[word_input, doc_pos_input, doc_neg_input], output=[merge_layer_, pos_layer_])

def ranking_loss(y_true, y_pred):
    pos = y_pred[:,0]
    neg = y_pred[:,1]
    loss = K.maximum(0.5 + neg - pos, 0.0)
    return K.mean(loss) + 0 * y_true

def dummy_loss(y_true, y_pred):
    # loss = K.max(y_pred) + 0 * y_true
    loss = y_pred + 0 * y_true
    return loss

model.compile(optimizer=Adam(lr=0.1), loss = {'merge_layer' : ranking_loss, "pos_layer": dummy_loss}, loss_weights=[1, 0])

print("finish model compiling")
print(model.summary())

target = np.array([9999] * len(word_data)) # useless since loss function make it times with 0
if os.path.exists(conf.path_checker):
    model.load_weights(conf.path_checker)

model.fit(
    {"word_idx":word_data, "doc_pos_idx":doc_pos_data, "doc_neg_idx":doc_neg_data},
    {"merge_layer":target, "pos_layer":target},
    batch_size=conf.batch_size,nb_epoch=conf.n_epoch,validation_split = 0.1,
    callbacks=[my_checker_point(doc_embed, word_embed, model, conf),
               # my_value_checker([word_embed_, doc_pos_embed_, doc_neg_embed_, pos_layer_, neg_layer_, merge_layer_]),
               ModelCheckpoint(filepath=conf.path_checker, verbose = 1, save_best_only=True)])