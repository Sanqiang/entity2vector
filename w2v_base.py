#word2vec base class
import json
import os.path
import os.path
import pickle
from collections import defaultdict, Counter

import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize

from stemmer import PorterStemmer
from functools import reduce
import re
from nltk.corpus import stopwords


class W2V_base:
    def default_idx(self):
        return -1

    def __init__(self, path, folder):
        self.tknzr = TweetTokenizer()
        # self.stknzr = sent_tokenize()
        self.stemmer = PorterStemmer()
        self.stops = set(stopwords.words("english"))

        self.path = path
        self.folder = folder

        # word based
        self.sample = 0.001
        self.min_count = 5;
        self.vocab_size = 0

        self.total_count = 0
        self.word_count = Counter()
        self.word2idx = defaultdict(self.default_idx)
        self.idx2word = {}
        self.word_sample = {}  # target word sample prob useless

        # entity based
        self.prod2idx = {}
        self.idx2prod = {}
        self.user2idx = {}
        self.idx2user = {}

        # train based
        self.batch_size = 300
        self.embedding_size = 128  # Dimension of the embedding vector.
        self.raw_sample_probs =  [0.4, 0.3, 0.15, 0.1, 0.05, 0.4, 0.3, 0.15, 0.1, 0.05]  # context word sample prob
        self.skip_window = len(self.raw_sample_probs)  # How many words to consider left and right.
        self.sample_probs = []
        sum = 0
        for prob in self.raw_sample_probs:
            sum += prob
            self.sample_probs.append(sum)
        #self.num_skips = 2  # How many times to reuse an input to generate a label.

        #self.num_negative_sampled = 64  # Number of negative examples to sample.

        self.data = []  # data set obj

        self.data_type = "yelp"
        self.file_encoding = "utf-8"
        # extension
        self.pos_mode = True

        self.get_stat()
        if self.pos_mode:
            self.interest_words = {}
            self.interest_tag = ["NOUN", "ADV", "ADJ"]
            self.get_stat_pos()
        self.batch_index = 0
        self.loop_index = 0

    def get_stat(self):
        filename = "/".join((self.folder, "stat"))
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

        # populate the count
        with open(self.path, "r", encoding=self.file_encoding) as ins:
            for line in ins:
                title, text, user, prod, rating = self.line_parser(line)
                # word based
                batch_text_data = " . ".join([batch_text_data, text, title])
                line_idx += 1
                if line_idx % 1000 == 0:
                    self.word_count += Counter(reduce(lambda x, y: x + y, self.parse(batch_text_data)))
                    batch_text_data = ""

        # out of loop word based
        self.word_count += Counter(reduce(lambda x, y: x + y, self.parse(batch_text_data)))  # not forget last batch

        # self.word_count = self.word_count.most_common(self.vocab_size)
        self.temp_word_count = Counter()
        for word in self.word_count:
            cnt = self.word_count[word]
            if cnt >= self.min_count:
                self.temp_word_count[word] = cnt
        self.word_count = self.temp_word_count
        del self.word_count["<UNK>"]
        self.vocab_size = len(self.word_count)

        #sort word
        self.word_count = self.word_count.most_common(self.vocab_size)

        # populate word2idx
        for word,cnt in self.word_count:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)  # -1 rather than 0 is discarded word
        # populate idx2word
        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))

        with open(self.path, "r", encoding=self.file_encoding) as ins:
            for line in ins:
                title, text, user, prod, rating = self.line_parser(line)
                # entity based note since we know the vocab size we can append entity embedding after that (first loop over data)
                if prod not in self.prod2idx:
                    prod_idx = self.vocab_size + len(self.prod2idx) + 1
                    self.prod2idx[prod] = prod_idx
                    self.idx2prod[prod_idx] = prod

        # populate data
        with open(self.path, "r", encoding=self.file_encoding) as ins:
            for line in ins:
                title, text, user, prod, rating = self.line_parser(line)
                text_data_idx = self.parse(" . ".join([text, title]), required_idx=True)

                obj = {"text_data": text_data_idx, "prod": prod, "user": user}

                self.data.append(obj)

                # note that we know both vocab size and entity size, we can append user embedding after that (second loop over data)
                if user not in self.user2idx:
                    user_idx = 1 + len(self.user2idx) + len(self.prod2idx) + self.vocab_size
                    self.user2idx[user] = user_idx
                    self.idx2user[user_idx] = user

        # calculate the sample
        # threshold_count = self.sample * self.total_count
        # for word in self.word_count:
        #    word_probability = (sqrt(self.w ord_count[word] / threshold_count) + 1) * (threshold_count / self.word_count[word])
        #    self.word_sample[word] = int(round(word_probability * 2**32))
        f = open(filename, 'wb')
        pickle_data = {"word2idx": self.word2idx, "idx2word": self.idx2word, "word_count": self.word_count,
                       "word_sample": self.word_sample, "total_count": self.total_count, "data": self.data,
                       "idx2user": self.idx2user, "user2idx": self.user2idx, "prod2idx": self.prod2idx,
                       "idx2prod": self.idx2prod}
        pickle.dump(pickle_data, f)

    def get_stat_pos(self):
        filename = "/".join((self.folder, "stat_pos"))
        if os.path.exists(filename):
            f = open(filename, 'rb')
            pickle_data = pickle.load(f)
            self.interest_words = pickle_data["interest_words"]
            return self.interest_words

        with open(self.path, "r", encoding=self.file_encoding) as ins:
            for line in ins:
                title, ttext, user, prod, rating = self.line_parser(line)
                text = " ".join((title, ttext))
                tagded_pairs = nltk.pos_tag(self.tknzr.tokenize(text), tagset='universal')
                for word, tag in tagded_pairs:
                    if tag in self.interest_tag:
                        stem_word = self.stemmer.get_stem_word(word)
                        if stem_word in self.word2idx:
                            self.interest_words[self.word2idx[stem_word]] = True
        f = open(filename, 'wb')
        pickle_data = {"interest_words": self.interest_words}
        pickle.dump(pickle_data, f)
        return self.interest_words

    # note that return format is title, text, user, prod, rating
    def line_parser(self, line):
        if self.data_type == "amz":
            obj = json.loads(line)
            reviewText = obj["reviewText"]
            summary = obj["summary"]
            reviewerID = obj["reviewerID"]
            overall = obj["overall"]
            asin = obj["asin"]
            return summary, reviewText, reviewerID, asin, overall
        elif self.data_type == "yelp":
            obj = json.loads(line)
            title = " "
            text = obj["text"]
            user = obj["user_id"]
            prod = obj["business_id"]
            rating = obj["stars"]
            return title, text, user, prod, rating

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

    def valid_word(self, word):
        if len(word) > 3:
            return True
        for idx in range(0, len(word)):
            if not str.isalpha(word[idx]):
                return False
        return True

    def token_transfer(self, token):
        if token in self.stops:
            return "<UNK>"
        token = re.sub("[^a-zA-Z]", "", token)
        token = token.lower()
        return token

    def parse(self, sents, required_idx = False):
        if required_idx:
            return [[self.word2idx[self.token_transfer(token)] for token in self.tknzr.tokenize(sent) if self.valid_word(token)] for
                sent in sent_tokenize(sents)]
        else:
            return [[self.token_transfer(token) for token in self.tknzr.tokenize(sent) if self.valid_word(token)] for
                sent in sent_tokenize(sents)]
