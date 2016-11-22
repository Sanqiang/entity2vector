from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.corpus import stopwords
from util.porter_stemmer import PorterStemmer
import nltk


class TextProcess:

    UNK = "<UNK>"

    tag_filter = ["NOUN","ADJ","ADP"]

    tknzr = TweetTokenizer()
    stemmer = PorterStemmer()

    @classmethod
    def initiliaze(self):
        return

    @classmethod
    def process_word(cls, word, tag, stem_flag = True, validate_flag = True, pos_filter = True, remove_stop_word = True):
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
            word = TextProcess.stemmer.stem(word,0,len(word)-1)
        return word

    @classmethod
    def process(cls, text):
        text = text.lower().replace("\n"," ").replace("\t"," ").replace("\v"," ")
        text_processed = ""
        for sent in sent_tokenize(text):
            for pair in nltk.pos_tag(TextProcess.tknzr.tokenize(sent), tagset='universal'):
                token = pair[0]
                tag = pair[1]
                text_processed = " ".join((text_processed, TextProcess.process_word(token,tag, stem_flag=True, validate_flag=True, pos_filter = True, remove_stop_word = True)))
            text_processed = "".join((text_processed,"\v"))
        return text_processed


if __name__ == '__main__':
    TextProcess.initiliaze()
    print("init")
    print(TextProcess.process("i eat djias but i do not like chicken"))