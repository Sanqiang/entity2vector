import os
import math
import operator

home = os.environ["HOME"]
path = "".join((home, "/data/yelp/review_processed_rest.txt"))

doc_by_word = {}

cnt_doc = {}
cnt_word = {}
cnt_word_total = 0
cnt_doc_total = 0

f = open(path)
for line in f:
    items = line.split("\t")
    if len(items) == 3:
        prod = items[0]
        tags = items[1]
        words = items[2]

        for word in words.split():
            if len(word) <= 2:
                continue
            if word not in cnt_word:
                cnt_word[word] = 0
            cnt_word_total += 1
            cnt_word[word]+=1

            if word not in doc_by_word:
                doc_by_word[word] = set()
            if prod not in doc_by_word[word]:
                doc_by_word[word].add(prod)

        if prod not in cnt_doc:
            cnt_doc[prod] = 0
        cnt_doc_total += 1
        cnt_doc[prod]+=1


weight_word = {}

f = open(path)
for line in f:
    items = line.split("\t")
    if len(items) == 3:
        prod = items[0]
        tags = items[1]
        words = items[2]

        for word in words.split():
            if len(word) <= 2:
                continue
            tf = cnt_word[word] / cnt_word_total
            idf = math.log(cnt_doc_total / len(doc_by_word[word]))
            if word not in weight_word:
                weight_word[word] = tf*idf

sorted_weight_word = sorted(weight_word.items(), key=operator.itemgetter(1))

idx = 1
for word, weight in sorted_weight_word:
    print(word,idx)
    idx += 1








