from nltk.tokenize import TweetTokenizer, sent_tokenize

from util.porter_stemmer import PorterStemmer


class TextProcess:

    UNK = "<UNK>"

    tknzr = TweetTokenizer()
    stemmer = PorterStemmer()

    @classmethod
    def process_word(cls, word, stem_flag = True, validate_flag = True):
        word = word.lower()
        if validate_flag:
            if len(word) <= 1 or len(word) >= 16:
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
            for token in TextProcess.tknzr.tokenize(sent):
                text_processed = " ".join((text_processed, TextProcess.process_word(token, stem_flag=False, validate_flag=False)))
            text_processed = "".join((text_processed,"\v"))
        return text_processed


#print(TextProcess.is_valid_word("http://www.yelp.com/biz/salsaritas-fresh-cantina-charlotte-5"))