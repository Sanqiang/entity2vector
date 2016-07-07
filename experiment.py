import os.path
import pickle
import random as rd
import sys
from collections import defaultdict, Counter, deque
from math import sqrt

import numpy as np
import tensorflow as tf
from nltk.tokenize import TweetTokenizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
from stemmer import PorterStemmer
from w2v_s import W2V_c

class Exp(W2V_c):
    def split(self):
        import json
        import heapq
        h = []
        with open(self.path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                #reviewText = obj["reviewText"]
                #summary = obj["summary"]
                #reviewerID = obj["reviewerID"]
                #overall = obj["overall"]
                #asin = obj["asin"]
                unixReviewTime = int(obj["unixReviewTime"])
                heapq.heappush(h, unixReviewTime)
        split = int(len(h) * 0.8)

        f_train = open("/".join((self.folder, "train.json")))
        f_test = open("/".join((self.folder, "test.json")))
        with open(self.path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                unixReviewTime = int(obj["unixReviewTime"])
                if unixReviewTime < split: # go to train
                    f_train.write(line)
                else:
                    f_test.write(line)

        return split

    def get_score_matrix(self):
        filename = "/".join((self.folder, "score"))
        if os.path.exists(filename):
            matrix = np.load(filename)
            return matrix
        import json
        prod_base_idx = min(self.prod2idx.values())
        user_base_idx = min(self.user2idx.values())
        prod_n = len(self.prod2idx)
        user_n = len(self.user2idx)

        matrix = lil_matrix((prod_n,user_n), dtype=np.float)

        with open(self.path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                #reviewText = obj["reviewText"]
                #summary = obj["summary"]
                reviewerID = obj["reviewerID"]
                overall = obj["overall"]
                asin = obj["asin"]
                prod_idx = self.prod2idx[asin] - int(prod_base_idx)
                user_idx = self.user2idx[reviewerID] - int(user_base_idx)
                matrix[prod_idx, user_idx] = overall
        np.save(filename, matrix)
        return matrix


    def test(self):
        embed = self.get_embedding(270000)
        prod_embed = embed[min(self.idx2prod.keys()):(1 + max(self.idx2prod.keys()))]
        nbrs = NearestNeighbors(n_neighbors=5, algorithm= "ball_tree").fit(prod_embed)

        matrix = self.get_score_matrix()
        nbrs2 = NearestNeighbors(n_neighbors=5, algorithm= "ball_tree").fit(matrix)

        return

    def get_embedding(self, step):
        path = "/".join((self.folder, "".join(("embedding_", str(step),"_np"))))
        if os.path.exists(path):
            embed = np.load(path)
            return embed

        graph = tf.Graph()
        with graph.as_default():

            valid_size = 16  # Random set of words to evaluate similarity on.
            valid_window = 100  # Only pick dev samples in the head of the distribution.
            valid_examples = 1 + np.random.choice(valid_window, valid_size, replace=False) #discard 0 which is discard word


            # Input data.
            train_inputs = tf.placeholder(tf.int32)
            train_labels = tf.placeholder(tf.int32)
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                embed_vocab_size = 1 + self.vocab_size + len(self.prod2idx) + len(self.user2idx)
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform([embed_vocab_size, self.embedding_size], -1.0, 1.0), name = "emb")
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([embed_vocab_size, self.embedding_size],
                                        stddev=1.0 / sqrt(self.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([embed_vocab_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                               self.num_negative_sampled, embed_vocab_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            saver = tf.train.Saver(max_to_keep = 2147483647)

            init = tf.initialize_all_variables()

        with tf.Session(graph=graph) as session:
            saver.restore(session, "/".join((self.folder, "".join(("embedding_", str(step))))))
            embed = session.run(embeddings)
            np.save(path, embed)
        return embed
exp = Exp("/home/sanqiang/Documents/data/Electronics_5.json", "amazon_electronics")
emb = exp.split()