import os.path
import pickle
import random as rd
import sys
from collections import defaultdict, Counter, deque
from math import sqrt

import numpy as np
import tensorflow as tf
from nltk.tokenize import TweetTokenizer

from stemmer import PorterStemmer


class W2V_c:
    def __init__(self, path, folder):
        self.tknzr = TweetTokenizer()
        self.stemmer = PorterStemmer()
        self.path =  path
        self.folder = folder

        #word based
        self.sample = 0.001
        self.vocab_size = 10000

        self.total_count = 0
        self.word_count = Counter()
        self.word2idx = defaultdict(int)
        self.idx2word = {}
        self.word_sample = {}  # target word sample prob useless

        #entity based
        self.prod2idx = {}
        self.idx2prod = {}
        self.user2idx = {}
        self.idx2user = {}

        #train based
        self.batch_size = 150
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.raw_sample_probs = [0.5, 0.3, 0.2] #context word sample prob
        self.skip_window = len(self.raw_sample_probs)  # How many words to consider left and right.
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

        filename =  "/".join((self.folder, "stat"))
        if os.path.exists(filename):
            f = open(filename, 'rb')
            obj = pickle.load(f)
            self.word2idx = obj["word2idx"]
            self.idx2word = obj["idx2word"]
            self.word_count = obj["word_count"]
            self.word_sample = obj["word_sample"]
            self.total_count = obj["total_count"]
            self.data = obj["data"]
            self.idx2prod = obj["idx2prod"]
            self.prod2idx = obj["prod2idx"]
            self.idx2user = obj["idx2user"]
            self.user2idx = obj["user2idx"]
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

                #word based
                batch_text_data = " ".join([batch_text_data, reviewText, summary])
                line_idx += 1
                if line_idx % 1000 == 0:
                    self.word_count += Counter(self.parse(batch_text_data))
                    batch_text_data = ""

                #entity based note since we know the vocab size we can append entity embedding after that (first loop over data)
                if asin not in self.prod2idx:
                    prod_idx = self.vocab_size + len(self.prod2idx) + 1
                    self.prod2idx[asin] = prod_idx
                    self.idx2prod[prod_idx] = asin

        #out of loop word based
        self.word_count += Counter(self.parse(batch_text_data)) #not forget last batch
        self.word_count = self.word_count.most_common(self.vocab_size)

        #populate word2idx
        for word, cnt in self.word_count:
            if word not in self.word2idx:
                self.word2idx[word] = 1 + len(self.word2idx) #0 is discarded word
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
                obj.pop("overall")

                self.data.append(obj)

                #note that we know both vocab size and entity size, we can append user embedding after that (second loop over data)
                if reviewerID not in self.user2idx:
                    user_idx = 1 + len(self.user2idx) + len(self.prod2idx) + self.vocab_size
                    self.user2idx[reviewerID] = user_idx
                    self.idx2user[user_idx] = reviewerID

        #calculate the sample
        #threshold_count = self.sample * self.total_count
        #for word in self.word_count:
        #    word_probability = (sqrt(self.w ord_count[word] / threshold_count) + 1) * (threshold_count / self.word_count[word])
        #    self.word_sample[word] = int(round(word_probability * 2**32))
        f = open(filename, 'wb')
        pickle_data = {"word2idx":self.word2idx,"idx2word":self.idx2word,"word_count":self.word_count,"word_sample":self.word_sample,"total_count":self.total_count, "data":self.data,
                       "idx2user":self.idx2user,"user2idx":self.user2idx,"prod2idx":self.prod2idx,"idx2prod":self.idx2prod}
        pickle.dump(pickle_data,f)

    def get_model(self, folder):
        if not os.path.exists(folder):
            return 0, None
        files = os.listdir(folder)
        max_n_step = 0
        max_file = ""
        for file in files:
            items = file.split("_")
            if len(items) >= 2:
                try:
                    n_step = int(items[-1])
                except ValueError:
                    continue
                if n_step > max_n_step:
                    max_n_step = n_step
                    max_file = file
        if max_n_step > 0:
            return max_n_step, "/".join((folder, max_file))
        else:
            return 0, None

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
            #overall = obj["overall"]
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

                    #add entity data
                    if self.prod2idx[asin] > self.vocab_size + len(self.prod2idx) + len(self.user2idx):
                        print("x")
                    if self.user2idx[reviewerID] > self.vocab_size + len(self.prod2idx) + len(self.user2idx):
                        print("x")
                    context_data.append(self.prod2idx[asin])
                    target_data.append(target_word)
                    context_data.append(self.user2idx[reviewerID])
                    target_data.append(target_word)

                #for next word
                buffer.append(text_data[word_idx])

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

            saver = tf.train.Saver()

            init = tf.initialize_all_variables()

        with tf.Session(graph=graph) as session:
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
                    print("Average loss at step ", step, ": ", average_loss)
                    filename = "_".join(["embedding",str(step)])
                    saver.save(session, "/".join((self.folder ,filename)), max_to_keep = sys.maxsize)
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
                            if nearest[k] < self.vocab_size:
                                close_word = self.idx2word[nearest[k]]
                                log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
        return


def main():
    model = W2V_c("/home/sanqiang/Documents/data/Electronics_5.json", "amazon_electronics")
    #model = W2V_c("/home/sanqiang/Documents/data/Amazon_Instant_Video_5.json", "amazon_instant_video")
    model.train(sys.maxsize)

main()

