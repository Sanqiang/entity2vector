from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.corpus import stopwords
from util.porter_stemmer import PorterStemmer
import nltk
import os


class TextProcess:

    UNK = "<UNK>"

    tag_filter = ["NOUN","ADJ","ADP"]

    tknzr = TweetTokenizer()
    stemmer = PorterStemmer()

    interested_words = set()

    stem_pairs = set()

    @classmethod
    def initiliaze(cls, path_pretraining):

        if path_pretraining is not None:

            f_pretraining = open(path_pretraining, "r")
            for line in f_pretraining:
                items = line.split(" ")
                word = items[0]
                TextProcess.interested_words.add(word)
        return

    @classmethod
    def process_word(cls, word, tag, stem_flag = True, validate_flag = True, pos_filter = True, remove_stop_word = True, only_interested_words = True):
        if pos_filter and tag not in TextProcess.tag_filter:
            return TextProcess.UNK
        word = word.lower()
        if remove_stop_word and word in stopwords.words('english'):
            return TextProcess.UNK
        if validate_flag:
            if len(word) <= 2 or len(word) >= 16:
                return TextProcess.UNK
            for ch in word:
                if not ch.isalpha():
                    return TextProcess.UNK
        if stem_flag:
            origin_word = word
            word = TextProcess.stemmer.stem(word,0,len(word)-1)
            stem_pair = (origin_word, word)
            TextProcess.stem_pairs.add(stem_pair)
        if word not in TextProcess.interested_words:
            return TextProcess.UNK
        return word

    @classmethod
    def generateStemPair(cls):
        home = os.environ["HOME"]
        path_stempairs = "".join((home, "/data/yelp/stem_pairs.txt"))
        f_stempairs = open(path_stempairs, "w")

        batch = ""
        for oword, nword in TextProcess.stem_pairs:
            line = "\t".join([oword, nword])
            line = "".join([line, "\n"])
            batch = "".join([batch, line])
            if len(batch) >= 1000000:
                f_stempairs.write(batch)
                batch = ""
        f_stempairs.write(batch)


    @classmethod
    def process(cls, text, pos_filter = False, stem_flag = False, validate_flag = True, remove_stop_word = True, only_interested_words = True):
        text = text.lower().replace("\n"," ").replace("\t"," ").replace("\v"," ")
        text_processed = ""
        for sent in sent_tokenize(text):
            if pos_filter:
                for pair in nltk.pos_tag(TextProcess.tknzr.tokenize(sent), tagset='universal'):
                    token = pair[0]
                    tag = pair[1]
                    nword = TextProcess.process_word(token, tag, stem_flag=stem_flag, validate_flag=validate_flag,
                                                                        pos_filter=pos_filter, remove_stop_word=remove_stop_word,
                                                                        only_interested_words=only_interested_words)
                    if nword is not TextProcess.UNK:
                        text_processed = " ".join((text_processed, nword))
                text_processed = "".join((text_processed,"\v"))
            else:
                for word in TextProcess.tknzr.tokenize(sent):
                    nword = TextProcess.process_word(word, None, stem_flag=stem_flag, validate_flag=validate_flag,
                                                                        pos_filter=pos_filter, remove_stop_word=remove_stop_word,
                                                                        only_interested_words=only_interested_words)
                    if nword is not TextProcess.UNK:
                        text_processed = " ".join((text_processed, nword))
        return text_processed.strip()


if __name__ == '__main__':
    import os
    home = os.environ["HOME"]
    path_pretraining = "".join((home, "/data/glove/glove.processed.twitter.27B.200d.txt"))
    TextProcess.initiliaze(path_pretraining)
    print("init")
    print(TextProcess.process("i eat djias but i do not like chicken"))