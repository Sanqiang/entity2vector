import os
from util.porter_stemmer import PorterStemmer
from util.text_process import TextProcess

home = os.environ["HOME"]
path = "".join((home, "/data/glove/glove.840B.300d.txt"))
path_processed = "".join((home, "/data/glove/glove.processed.840B.300d.txt"))



f = open(path, "r")
f_processed = open(path_processed, "w")
batch = ""
words = set()
for line in f:
    items = line.split(" ")
    items[0] = TextProcess.process_word(items[0])
    if items[0] not in words:
        words.add(items[0])
        batch = "".join((batch, " ".join(items), ""))
        if len(batch) >= 100000:
            f_processed.write(batch)
            batch = ""

f_processed.write(batch)
