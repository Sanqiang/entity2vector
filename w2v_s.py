#word2vec simple training
import random as rd
import sys
from collections import deque
from math import sqrt

import numpy as np
#import tensorflow as tf

from w2v_base import W2V_base

class W2V_c(W2V_base):
    def __init__(self, path, folder):
        W2V_base.__init__(self, path, folder)

    def get_batch(self):
        #prepare buffer
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = deque(maxlen=span)

        context_data = [] #list append is better than numpy append, so using list append first and then convert into numpy obj
        target_data = [] # same as above
        for i in range(0, self.batch_size):
            idx = (self.batch_index + i) % len(self.data)
            self.loop_index = int((self.batch_index + i) / len(self.data))
            obj = self.data[idx]
            user = obj["user"]
            #overall = obj["overall"]
            prod = obj["prod"]
            text_data = obj["text_data"]

            for text_data_sent in text_data:
                word_idx = 0
                while len(buffer) < span and word_idx < len(text_data_sent):
                    buffer.append(text_data_sent[word_idx])
                    word_idx += 1
                word_idx -= 1

                for word_idx in range(word_idx, len(text_data_sent)):

                    target_idx = int((len(buffer)+1)/2)
                    target_word = buffer[target_idx] #consider buffer is shorter than  self.skip_window
                    context_word = target_word
                    avoid_context_word = [target_word]

                    for cnt_idx in range(0, self.skip_window): #random pick up skip_window context word
                        reset = self.skip_window * 5
                        while context_word in avoid_context_word: #for avoid repeat sample
                            r = rd.random()
                            reset -= 1
                            if reset < 0:
                                break
                            for rank_idx in range(0, self.skip_window): #from closest to farest
                                if r <= self.sample_probs[rank_idx]:
                                    if r >= 0.5:
                                        if target_idx - (rank_idx + 1) > 0:
                                            context_word = buffer[target_idx - (rank_idx + 1)]
                                    else:
                                        if target_idx + (rank_idx + 1) < len(buffer):
                                            context_word = buffer[target_idx + (rank_idx + 1)]
                                    break
                        if context_word not in avoid_context_word:
                            avoid_context_word.append(context_word)
                            context_data.append(context_word)
                            target_data.append(target_word)

                        #add entity data
                        context_data.append(self.prod2idx[prod])
                        target_data.append(target_word)
                        context_data.append(self.user2idx[user])
                        target_data.append(target_word)

                        if self.pos_mode:
                            if target_word in self.interest_words:
                                target_data.append(self.prod2idx[prod])
                                context_data.append(target_word)
                        else:
                            target_data.append(self.prod2idx[prod])
                            context_data.append(target_word)
                            #target_data.append(self.user2idx[user])
                            #context_data.append(target_word)
                    #for next word
                    buffer.append(text_data_sent[word_idx])

        #update global batch_index
        self.batch_index += self.batch_size

        context_data =  np.array(context_data) #np.ndarray(shape=(len(context_data)), dtype=np.int32)
        target_data =  np.array(target_data)[np.newaxis] #np.ndarray(shape=(len(target_data), 1), dtype=np.int32)
        return context_data, target_data.T

    def train(self, num_steps):
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

        with tf.Session(graph=graph, config=tf.ConfigProto(intra_op_parallelism_threads=20, inter_op_parallelism_threads = 20)) as session:
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")

            model_step, model_file = self.get_model(self.folder)
            if model_file is not None:
                saver.restore(session, model_file)

            average_loss = 0
            for step in range(model_step, model_step + num_steps):

                batch_inputs, batch_labels = self.get_batch()
                #print("current step", step)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 50
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss, " current loop", self.loop_index)
                    filename = "_".join(["embedding",str(step)])
                    saver.save(session, "/".join((self.folder ,filename)))
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):

                        valid_word = self.idx2word[valid_examples[i]]
                        top_k = 10  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in range(top_k):
                            if nearest[k] in self.idx2word:
                                close_word = self.idx2word[nearest[k]]
                                log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
        return


def main():
    model = W2V_c("/home/sanqiang/Documents/data/Electronics_5.json", "amazon_electronics_nmodel")
    #model = W2V_c("/home/sanqiang/Documents/data/Amazon_Instant_Video_5.json", "amazon_instant_video")
    model.train(sys.maxsize)

#main()

