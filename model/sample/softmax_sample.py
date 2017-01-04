import numpy as np
import tensorflow as tf
sess = tf.Session()
x = tf.constant(np.array([[.1,.2,.3,.4],[.1,.2,.3,.4]]))
print(sess.run(tf.nn.softmax(x)))
# softmax runs in horizontal