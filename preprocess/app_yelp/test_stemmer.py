from util.text_process import TextProcess

word = "yummy"

stem_word = TextProcess.stemmer.stem(word, 0, len(word)-1)
print(stem_word)