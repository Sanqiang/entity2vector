import os
from text_process import TextProcess
import json

home = os.environ["HOME"]
path = "".join((home, "/data/yelp/review.json"))
path_processed = "".join((home, "/data/yelp/review_processed.json"))


f = open(path, "r")
f_processed = open(path_processed, "w")
for line in f:
    obj = json.loads(line)
    text = TextProcess.process(obj["text"])
    user_id = obj["user_id"]
    business_id = obj["business_id"]
    stars = obj["stars"]
    obj_processed = {"text":text,"user_id":user_id,"business_id":business_id,"stars":stars}
    line_processed = json.dumps(obj_processed)
    f_processed.write(line_processed)
    f_processed.write("\n")

