import json
import os

from util.text_process import TextProcess

home = os.environ["HOME"]
path = "".join((home, "/data/yelp/review.json"))
path_processed = "".join((home, "/data/yelp/review_processed_nostem_rest.txt"))

filter_rest = True
prods = set()
tags = set()
if filter_rest:
    path_prod = "".join((home, "/data/yelp/business.json"))
    f_prod = open(path_prod, "r")
    for line in f_prod:
        obj = json.loads(line)
        business_id = str(obj["business_id"])
        categories = obj["categories"]
        if "Restaurants" in categories:
            prods.add(business_id)
            for category in categories:
                tags.add(category)

    for tag in tags:
        print(tag)


f = open(path, "r")
f_processed = open(path_processed, "w")
batch = ""
for line in f:
    obj = json.loads(line)
    text = TextProcess.process(obj["text"])
    user_id = str(obj["user_id"])
    business_id = str(obj["business_id"])
    stars = str(obj["stars"])

    if not filter_rest or business_id in prods:
        # line_processed = "\t".join((user_id, business_id, stars, text))
        line_processed = "\t".join((business_id, text))
        batch = "\n".join((batch, line_processed))
        if len(batch) > 100000:
            f_processed.write(batch)
            f_processed.write("\n")
            batch = ""
f_processed.write(batch)
f_processed.write("\n")

