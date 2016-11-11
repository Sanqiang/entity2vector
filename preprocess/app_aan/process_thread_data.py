import random
import os

from util.text_process import TextProcess

k = 8

home = os.environ["HOME"]
path_processed = "".join((home, "/data/aan/paper_processed_nostem"))
path_processed_origin = "".join((path_processed, ".txt"))
fs = []


for i in range(0, k):
    path_temp = "".join((path_processed,"_", str(i),".txt"))
    fs.append(open(path_temp, "w"))


f = open(path_processed_origin, "r")
f_processed = open(path_processed, "w")
for line in f:
    i = random.randint(0, k-1)
    fs[i].write(line)



