#word2vec pretraining imple
from stemmer import PorterStemmer
import json
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
import os.path


class Vector_Process:
    def __init__(self, folder, file_origin, file_update, vector_length, words):
        self.folder = folder
        self.file_origin = file_origin
        self.file_update = file_update
        self.vector_length = vector_length
        #self.stemmer = PorterStemmer()
        self.words = words

    def process(self):
        path_origin = "/".join((self.folder, self.file_origin))
        path_update = "/".join((self.folder, self.file_update))
        f_update = open(path_update, "w")
        f_origin = open(path_origin, "r")
        str_update = ""
        for line in f_origin:
            items = line.split(" ")
            word = items[0]
            if word not in self.words:
                continue
            nline = " ".join(items)
            str_update = "\n".join((str_update, nline))
        f_update.write(str_update)
        f_origin.close()
        f_update.close()

    def check_word(self):
        return False


class Pair_Process():
    def __init__(self, folder, file_origin, file_pair, prod_sign=False, usr_sign=False, pos_sign=False):
        self.folder = folder
        self.file_origin = file_origin
        self.file_pair = file_pair
        #self.stemmer = PorterStemmer()
        self.words = set()
        self.intersted_pos = ["NOUN", "ADV", "ADJ"]
        self.prod_sign = prod_sign
        self.usr_sign = usr_sign
        self.pos_sign = pos_sign

    def line_parser(self, line):
        obj = json.loads(line)
        title = " "
        context = obj["text"]
        user = obj["user_id"]
        prod = obj["business_id"]
        rating = obj["stars"]
        return title, context, user, prod, rating

    def populate_words(self, file_word):
        path_words = "/".join((self.folder, file_word))
        if os.path.exists(path_words):
            f_words = open(path_words, "r")
            for word in f_words:
                self.words.add(word)
            return self.words

        path_origin = "/".join((self.folder, self.file_origin))
        f_origin = open(path_origin, "r")
        for line in f_origin:
            title, context, user, prod, rating = self.line_parser(line)
            text = ". ".join((title, context))
            text = [[token for token in word_tokenize(sent)] for sent in sent_tokenize(text)]
            for sent in text:
                sent = nltk.pos_tag(sent, tagset='universal')
                for word,tag in sent:
                    if self.pos_sign and tag not in self.intersted_pos:
                        continue
                    if self.prod_sign:
                        self.words.add(word)
                    if self.usr_sign:
                        self.words.add(word)

        f_words = open(path_words, "w")
        for word in self.words:
            f_words.write(word)
            f_words.write("\n")
        return self.words


    def process(self):
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
                sent = nltk.pos_tag(sent, tagset='universal')
                for word, tag in sent:
                    if word in self.words:
                        if self.pos_sign and tag not in self.intersted_pos:
                            continue
                        if self.prod_sign:
                            results.append((prod, word))
                            n_prod += 1
                        if self.usr_sign:
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
        print("#user is", n_user)
        print("#prod is", n_prod)
        print("#pair is", n_pair)


def main():
    pp = Pair_Process("/home/sanqiang/data/yelp", "NVu.json", "pair_NVu", pos_sign=True, prod_sign=True)
    words = pp.populate_words("word.txt")
    print("1")

    vp = Vector_Process("/home/sanqiang/data/glove","glove.twitter.27B.200d.txt","glove.twitter.27B.200d.update.txt",200, words)
    vp.process()
    print("2")

    pp.process()
    print("3")

main()

