#word2vec pretraining imple
from stemmer import PorterStemmer
import json
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize


class Vector_Process:
    def __init__(self, folder, file_origin, file_update, vector_length):
        self.folder = folder
        self.file_origin = file_origin
        self.file_update = file_update
        self.vector_length = vector_length
        self.stemmer = PorterStemmer()
        self.words = set()

    def process(self):
        path_origin = "/".join((self.folder, self.file_origin))
        path_update = "/".join((self.folder, self.file_update))
        f_origin = open(path_origin, "r")
        str_update = ""
        for line in f_origin:
            items = line.split(" ")
            word = self.stemmer.get_stem_word(items[0])
            items[0] = word
            self.words.add(word)
            nline = " ".join(items)
            str_update = "\n".join((str_update, nline))
        f_update = open(path_update, "w")
        f_update.write(str_update)
        f_origin.close()
        f_update.close()

    def check_word(self):
        return False


class Pair_Process():
    def __init__(self, folder, file_origin, file_pair, words):
        self.folder = folder
        self.file_origin = file_origin
        self.file_pair = file_pair
        self.stemmer = PorterStemmer()
        self.words = words
        self.intersted_pos = ["NOUN", "ADV", "ADJ"]

    def line_parser(self, line):
        obj = json.loads(line)
        title = " "
        context = obj["text"]
        user = obj["user_id"]
        prod = obj["business_id"]
        rating = obj["stars"]
        return title, context, user, prod, rating

    def process(self, prod_sign=False, usr_sign=False, pos_sign=False):
        n_user = 0
        n_prod = 0
        n_pair = 0
        path_origin = "/".join((self.folder, self.file_origin))
        path_pair = "/".join((self.folder, self.file_pair))
        results = []
        f_origin = open(path_origin, "r")
        f_update = open(path_pair, "w")
        for line in f_origin:
            title, context, user, prod, rating = self.line_parser(line)
            text = ". ".join((title, context))
            text = [[token for token in word_tokenize(sent)] for sent in sent_tokenize(text)]
            for sent in text:
                for word in sent:
                    word = self.stemmer.get_stem_word(word)
                    if word in self.words:
                        if pos_sign and word not in self.intersted_pos:
                            continue
                        if prod_sign:
                            results.append((prod, word))
                            n_prod += 1
                        if usr_sign:
                            results.append((user, word))
                            n_user += 1
            if len(results) >= 10000:
                n_pair += len(results)
                print(len(results))
                for entity, word in results:
                    f_update.write(entity)
                    f_update.write(" ")
                    f_update.write(word)
                results = []
        n_pair += len(results)
        print(len(results))
        for entity, word in results:
            f_update.write(entity)
            f_update.write(" ")
            f_update.write(word)
        print("#user is", n_user, "\n")
        print("#prod is", n_prod, "\n")
        print("#pair is", n_pair, "\n")

def main():
    vp = Vector_Process("/home/sanqiang/data/glove","glove.twitter.27B.200d.txt","glove.twitter.27B.200d.update.txt",200)
    vp.process()

    pp = Pair_Process("/home/sanqiang/data/yelp", "review.json", "pair_review", vp.words)
    pp.process(pos_sign=True, prod_sign=True)

main()

