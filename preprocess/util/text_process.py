from nltk.tokenize import TweetTokenizer, sent_tokenize

from util.porter_stemmer import PorterStemmer


class TextProcess:

    UNK = "<UNK>"

    tknzr = TweetTokenizer()
    stemmer = PorterStemmer()

    @classmethod
    def process_word(cls, word):
        word = word.lower()
        if len(word) <= 1 or len(word) >= 16:
            return TextProcess.UNK
        for ch in word:
            if not ch.isalpha():
                return TextProcess.UNK
        word = TextProcess.stemmer.stem(word,0,len(word)-1)

        return word

    @classmethod
    def process(cls, text):
        text = text.lower().replace("\n"," ").replace("\t"," ").replace("\v"," ")
        text_processed = ""
        for sent in sent_tokenize(text):
            for token in TextProcess.tknzr.tokenize(sent):
                text_processed = " ".join((text_processed, TextProcess.process_word(token)))
            text_processed = "".join((text_processed,"\v"))
        return text_processed


#print(TextProcess.is_valid_word("http://www.yelp.com/biz/salsaritas-fresh-cantina-charlotte-5"))