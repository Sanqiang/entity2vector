import pickle
import random as rd
from collections import defaultdict, Counter, deque
from math import sqrt

import numpy as np
import tensorflow as tf
from nltk.tokenize import TweetTokenizer

from stemmer import PorterStemmer
from nltk.corpus import stopwords


class W2V_c:
    def __init__(self, path):
        self.tknzr = TweetTokenizer()
        self.stemmer = PorterStemmer()
        self.path =  path

        self.sample = 0.001
        self.vocab_size = 10000

        self.total_count = 0
        self.word_count = Counter()
        self.word2idx = defaultdict(int)
        self.idx2word = {}

        self.word_sample = {} #target word sample prob useless

        self.batch_size = 5
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.skip_window = 3  # How many words to consider left and right.
        self.raw_sample_probs = [0.5, 0.3, 0.2] #context word sample prob
        self.sample_probs = []
        sum = 0
        for prob in self.raw_sample_probs:
            sum += prob
            self.sample_probs.append(sum)
        self.num_skips = 2  # How many times to reuse an input to generate a label.

        self.num_negative_sampled = 64  # Number of negative examples to sample.

        self.data = [] #data set obj

        self.get_stat()

        self.batch_index = 0


    def valid_word(self, word):
        if len(word) > 3:
            return True
        for idx in range(0, len(word)):
            if not str.isalpha(word[idx]):
                return False
        return True

    def parse(self, sent):
        return [self.stemmer.get_stem_word(token) for token in self.tknzr.tokenize(sent) if self.valid_word(token)]

    def get_stat(self):
        import json
        import collections
        import os.path

        filename = "stat"
        if os.path.exists(filename):
            f = open(filename, 'rb')
            obj = pickle.load(f)
            self.word2idx = obj["word2idx"]
            self.idx2word = obj["idx2word"]
            self.word_count = obj["word_count"]
            self.word_sample = obj["word_sample"]
            self.total_count = obj["total_count"]
            self.data = obj["data"]
            return

        line_idx = 0
        batch_text_data = ""

        #populate the count
        with open(self.path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                reviewText = obj["reviewText"]
                summary = obj["summary"]
                reviewerID = obj["reviewerID"]
                overall = obj["overall"]
                asin = obj["asin"]
                batch_text_data = " ".join([batch_text_data, reviewText, summary])
                line_idx += 1
                if line_idx % 1000 == 0:
                    self.word_count += collections.Counter(self.parse(batch_text_data))
                    batch_text_data = ""
        self.word_count += collections.Counter(self.parse(batch_text_data)) #not forget last batch
        self.word_count = self.word_count.most_common(self.vocab_size-1)

        #populate word2idx
        for word, cnt in self.word_count:
            if word not in self.word2idx:
                self.word2idx[word] = 1+len(self.word2idx) #0 is discarded word
        #populate idx2word
        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))

        #populate data
        with open(self.path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                reviewText = obj["reviewText"]
                summary = obj["summary"]
                reviewerID = obj["reviewerID"]
                overall = obj["overall"]
                asin = obj["asin"]
                text_data = self.parse(" ".join([reviewText, summary]))
                text_data_idx = []
                for word in text_data:
                    text_data_idx.append(self.word2idx[word])

                obj["text_data"] = text_data_idx
                #for save memory
                obj.pop("reviewText")
                obj.pop("summary")

                self.data.append(obj)

        #calculate the sample
        #threshold_count = self.sample * self.total_count
        #for word in self.word_count:
        #    word_probability = (sqrt(self.w ord_count[word] / threshold_count) + 1) * (threshold_count / self.word_count[word])
        #    self.word_sample[word] = int(round(word_probability * 2**32))
        f = open(filename, 'wb')
        pickle.dump({"word2idx":self.word2idx,"idx2word":self.idx2word,"word_count":self.word_count,"word_sample":self.word_sample,"total_count":self.total_count, "data":self.data},f)

    def get_batch(self):
        #prepare buffer
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = deque(maxlen=span)

        context_data = [] #list append is better than numpy append, so using list append first and then convert into numpy obj
        target_data = [] # same as above
        for i in range(0, self.batch_size):
            idx = (self.batch_index + i) % len(self.data)
            obj = self.data[idx]

            reviewerID = obj["reviewerID"]
            overall = obj["overall"]
            asin = obj["asin"]
            text_data = obj["text_data"]

            word_idx = 0
            while len(buffer) < span:
                buffer.append(text_data[word_idx])
                word_idx += 1
                if word_idx >= len(text_data):
                    break

            for word_idx in range(word_idx, len(text_data)):

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

                #for next word
                buffer.append(text_data[word_idx])

        #update global batch_index
        self.batch_index += self.batch_size

        context_data =  np.array(context_data) #np.ndarray(shape=(len(context_data)), dtype=np.int32)
        target_data =  np.array(target_data)[np.newaxis] #np.ndarray(shape=(len(target_data), 1), dtype=np.int32)
        return context_data, target_data.T

    def train(self):
        embedding_size = 128  # Dimension of the embedding vector.
        graph = tf.Graph()

        with graph.as_default():

            valid_size = 16  # Random set of words to evaluate similarity on.
            valid_window = 100  # Only pick dev samples in the head of the distribution.
            valid_examples = np.random.choice(valid_window, valid_size, replace=False)


            # Input data.
            train_inputs = tf.placeholder(tf.int32)
            train_labels = tf.placeholder(tf.int32)
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                    tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0), name = "emb")
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocab_size, embedding_size],
                                        stddev=1.0 / sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                               self.num_negative_sampled, self.vocab_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            saver = tf.train.Saver()

            init = tf.initialize_all_variables()

        self.num_steps = 1000000




        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")

            average_loss = 0
            for step in range(self.num_steps):

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
                    print("Average loss at step ", step, ": ", average_loss)
                    filename = "_".join(["embedding",str(step)])
                    saver.save(session, filename)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):

                        valid_word = self.idx2word[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in range(top_k):
                            close_word = self.idx2word[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
        return


def main():
    model = W2V_c("/home/sanqiang/Documents/data/Electronics_5.json")
    model.train()

main()

