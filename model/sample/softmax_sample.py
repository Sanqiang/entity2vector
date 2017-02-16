import numpy as np
import tensorflow as tf
import keras as K

def relu_max(x):
    e = K.relu(x, alpha=0, max_value=None)
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s

sess = tf.Session()
x = tf.constant(np.array([[-.1,.2,.3,.4],[.1,.2,.3,.4]]))
print(sess.run(tf.nn.softmax(x)))
print(sess.run(tf.nn.relu(x)))
# softmax runs in horizontal