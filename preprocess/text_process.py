import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.corpus import stopwords

class TextProcess:

    tknzr = TweetTokenizer()

    @classmethod
    def process(cls, text):
        text = text.lower().replace("\n"," ").replace("\t"," ").replace("\v"," ")
        text_processed = ""
        for sent in sent_tokenize(text):
            for token in TextProcess.tknzr.tokenize(sent):
                text_processed = " ".join((text_processed, token))
            text_processed = "".join((text_processed,"\v"))
        return text_processed

