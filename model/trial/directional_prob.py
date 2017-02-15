import json
import os
from nltk.tokenize import word_tokenize, sent_tokenize

word_target = "ribeye"
word_condition = "steak"

up = 1
down = 1

home = os.environ["HOME"]
path = "".join([home, "/data/yelp/review.json"])
path2 = "".join([home, "/data/yelp/review2.json"])
f = open(path, "r")
f2 = open(path2, "w")
batch = ""
for line in f:
    obj = json.loads(line)
    text = obj["text"]


    for sent in sent_tokenize(text):
        contain_target = False
        contain_condition = False
        for word in word_tokenize(sent):
            if word == word_target:
                contain_target = True
            if word == word_condition:
                contain_condition = True
            batch = " ".join([batch, word])
        if contain_condition and contain_target:
            up += 1
        if contain_condition:
            down += 1
        batch = "".join([batch, "\t"])
    batch = "".join([batch, "\n"])
    if len(batch) >= 10000:
        print(len(batch))
        f2.write(batch)
        batch = ""

f2.write(batch)
prob = up / down
print(up)
print(down)
print(prob)
