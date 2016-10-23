from gensim.models.word2vec import Word2Vec
import os

home = os.environ["HOME"]
path_processed = "".join((home, "/data/glove/glove.processed.840B.300d.txt"))

model = Word2Vec.load_word2vec_format(path_processed)
print("finished init")
while True:
    word = input()
    words = model.most_similar(word)
    print(words)
    print("=======")