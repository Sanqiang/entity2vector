import json
import os

from util.text_process import TextProcess

home = os.environ["HOME"]
path_processed = "".join((home, "/data/yelp/review_processed_0.txt"))

counts = {}


f_processed = open(path_processed, "r")
for line in f_processed:
    items = line.split("\t")
    if items[0] not in counts:
        counts[items[1]] = set()
    for w in items[1].split(" "):
        counts[items[1]].add(w)

for w in counts:
    print(len(counts[w]))

