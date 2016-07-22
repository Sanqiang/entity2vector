#word2vec old imple
from collections import defaultdict, Counter
from six import iteritems
from math import sqrt
import numpy as np
import tensorflow as tf
from stemmer import PorterStemmer
import pickle
import json
import linecache
import collections
import numpy as np
import random as rd

class W2V:
    def __init__(self, path):
        self.stemmer = PorterStemmer()
        self.path =  path

        self.cur_idx = 0
        self.batch = 2
        self.sample = 0.001
        self.vocab_size = 10000

        self.total_count = 0
        self.word_count = Counter()
        self.word2idx = defaultdict(int)
        self.idx2word = {}
        self.word_sample = {}

        self.batch_size = 128
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.skip_window = 3  # How many words to consider left and right.
        self.raw_sample_probs = [0.5, 0.3, 0.2]
        self.sample_probs = []
        sum = 0
        for prob in self.raw_sample_probs:
            sum += prob
            self.sample_probs.append(sum)
        self.num_skips = 2  # How many times to reuse an input to generate a label.

        self.valid_size = 16  # Random set of words to evaluate similarity on.
        self.valid_window = 100  # Only pick dev samples in the head of the distribution.
        #self.valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        self.num_negative_sampled = 64  # Number of negative examples to sample.

        self.batch_index = 0
        self.get_stat()

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
            return

        line_idx = 0
        text_data = ""

        with open(self.path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                reviewText = obj["reviewText"]
                summary = obj["summary"]
                reviewerID = obj["reviewerID"]
                overall = obj["overall"]
                asin = obj["asin"]
                text_data = " ".join([text_data ,reviewText, summary])

                line_idx += 1
                if line_idx % 1000 == 0:
                    self.word_count + collections.Counter(self.stemmer.get_stem_wordlist(text_data.split()))
                    text_data = ""

        self.word_count = self.word_count.most_common(self.vocab_size-1)

        for word,cnt in self.word_count:
            self.word2idx[word] = 1+len(self.word2idx)
        #self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))

        #calculate the sample
        #threshold_count = self.sample * self.total_count
        #for word in self.word_count:
        #    word_probability = (sqrt(self.word_count[word] / threshold_count) + 1) * (threshold_count / self.word_count[word])
        #    self.word_sample[word] = int(round(word_probability * 2**32))
        f = open(filename, 'wb')
        pickle.dump({"word2idx":self.word2idx,"idx2word":self.idx2word,"word_count":self.word_count,"word_sample":self.word_sample,"total_count":self.total_count},f)

    def get_batch(self):


        #global batch_index

        #calculate sample prob inside window
        #sample_probs = []
        #sum = (1<<(self.skip_window)) - 1
        #prob = 0
        #for idx in range(self.skip_window,0,-1):
        #    prob += (1<<(idx-1)) / sum
        #    sample_probs.append(prob)

        #prepare buffer
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        context_data = [] #list append is better than numpy append, so using list append first and then convert into numpy obj
        target_data = [] # same as above
        for i in range(1, self.batch_size):
            idx = self.batch_index + i
            #line = linecache.getline(self.path, idx)
            line = '{ "reviewerID": "A2SUAM1J3GNN3B", "asin": "0000013714", "reviewerName": "J. McDonald", "helpful": [2, 3], "reviewText": "I bought this for my husband who plays the piano. He is having a wonderful time playing these old hymns. The music is at times hard to read because we think the book was published for singing from more than playing from. Great purchase though!", "overall": 5.0, "summary": "Heavenly Highway Hymns", "unixReviewTime": 1252800000, "reviewTime": "09 13, 2009" }'

            if line is None or len(line) == 0:
                print("current idx,", idx, " current batch_idx, ", self.batch_index, " line: ", line)
                continue

            obj = json.loads(line)
            reviewText = obj["reviewText"]
            summary = obj["summary"]
            reviewerID = obj["reviewerID"]
            overall = obj["overall"]
            asin = obj["asin"]
            text_data = " ".join([reviewText, summary]).split()

            for word_idx in range(0, len(text_data)):
                while len(buffer) < span:
                    buffer.append(self.word2idx[self.stemmer.get_stem_word(text_data[word_idx])])

                target_word = self.word2idx[buffer[self.skip_window]]
                context_word = target_word
                avoid_context_word = [target_word]
                r = rd.random()
                for cnt_idx in range(0, int(self.skip_window/2)): #random pick up skip_window/2 context word
                    while context_word in avoid_context_word: #for avoid repeat sample
                        for rank_idx in range(0,self.skip_window): #from closest to farest
                            if r <= self.sample_probs[rank_idx]:
                                if rd.random() >= 0.5:
                                    context_word = self.word2idx[buffer[self.skip_window - (rank_idx + 1)]]
                                else:
                                    context_word = self.word2idx[buffer[self.skip_window + (rank_idx + 1)]]
                                break
                    if context_word not in avoid_context_word:
                        avoid_context_word.append(context_word)
                        context_data.append(context_word)
                        target_data.append(target_word)

                #for next word
                buffer.append(self.stemmer.get_stem_word(text_data[word_idx]))

        #update global batch_index
        self.batch_index += self.batch_size

        context_data = np.ndarray(shape=(len(context_data)), dtype=np.int32)
        target_data = np.ndarray(shape=(len(target_data), 1), dtype=np.int32)
        return context_data, target_data

    def train(self):
        embedding_size = 128  # Dimension of the embedding vector.
        graph = tf.Graph()

        with graph.as_default():

            # Input data.
            train_inputs = tf.placeholder(tf.int32)
            train_labels = tf.placeholder(tf.int32)

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

        return


def main():
    model = W2V("/Users/zhaosanqiang916/Data/reviews_Books.json")
    model.train()

main()

