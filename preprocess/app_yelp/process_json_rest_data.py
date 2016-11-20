import json
import os


home = os.environ["HOME"]
path = "".join((home, "/data/yelp/review.json"))
path_rest = "".join((home, "/data/yelp/review_rest.json"))

prods = set()
prod_tag = {}

path_prod = "".join((home, "/data/yelp/business.json"))
f_prod = open(path_prod, "r")
for line in f_prod:
    obj = json.loads(line)
    business_id = str(obj["business_id"])
    categories = obj["categories"]
    if "Restaurants" in categories:
        prods.add(business_id)
        if not business_id in prod_tag:
            prod_tag[business_id] = set()
        for category in categories:
            if "Restaurants" != category:
                prod_tag[business_id].add(category.replace(" ",""))


f = open(path, "r")
f_rest = open(path_rest, "w")
batch = ""
for line in f:
    obj = json.loads(line)
    user_id = str(obj["user_id"])
    business_id = str(obj["business_id"])
    stars = str(obj["stars"])

    if business_id in prods:
        batch = "".join((batch, line))
        if len(batch) > 100000:
            f_rest.write(batch)
            batch = ""
f_rest.write(batch)