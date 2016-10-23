import json
import os

from util.text_process import TextProcess

home = os.environ["HOME"]
path = "".join((home, "/data/yelp/review.json"))
path_processed = "".join((home, "/data/yelp/review_processed.txt"))


f = open(path, "r")
f_processed = open(path_processed, "w")
for line in f:
    obj = json.loads(line)
    text = TextProcess.process(obj["text"])
    user_id = str(obj["user_id"])
    business_id = str(obj["business_id"])
    stars = str(obj["stars"])

    # line_processed = "\t".join((user_id, business_id, stars, text))
    line_processed = "\t".join((business_id, text))
    f_processed.write(line_processed)
    f_processed.write("\n")

