import json
import os

from util.text_process import TextProcess

home = os.environ["HOME"]
path = "".join((home, "/data/yelp/review.json"))

filter_rest = True
prods = set()
prod_tag = {}
prod_loc = {}
if filter_rest:
    path_prod = "".join((home, "/data/yelp/business.json"))
    f_prod = open(path_prod, "r")
    for line in f_prod:
        obj = json.loads(line)
        business_id = str(obj["business_id"])
        categories = obj["categories"]
        state = obj["state"]
        city = obj["city"].replace(" ","")
        loc = "_".join((state,city))
        if "Restaurants" in categories:
            prods.add(business_id)
            if not business_id in prod_tag:
                prod_tag[business_id] = set()
            for category in categories:
                prod_tag[business_id].add(category.replace(" ",""))
            prod_loc[business_id] = loc;



f = open(path, "r")
user_loc_cnt = {}
user_loc_prob = {}
user_cnt_max = {}
user_prob_max = {}
batch = ""
for line in f:
    obj = json.loads(line)
    business_id = str(obj["business_id"])

    if business_id in prods:
        user_id = str(obj["user_id"])
        stars = str(obj["stars"])
        loc = prod_loc[business_id]

        if user_id not in user_loc_cnt:
            user_loc_cnt[user_id] = {}
        loc_stat = user_loc_cnt[user_id]
        if loc not in loc_stat:
            loc_stat[loc] = 0
        loc_stat[loc]+=1

for user_id in user_loc_cnt.keys():

    cnt = 0
    locs = user_loc_cnt[user_id]

    #process prob
    user_loc_prob[user_id] = {}
    temp = user_loc_prob[user_id]

    #process max
    if user_id not in user_cnt_max:
        user_cnt_max[user_id] = 0

    if user_id not in user_prob_max:
        user_prob_max[user_id] = 0

    for loc in locs.keys():
        cnt += locs[loc]
    for loc in locs.keys():
        temp[loc] = locs[loc] / cnt
        if temp[loc] > user_prob_max[user_id]:
            user_prob_max[user_id] = temp[loc]
        if locs[loc] > user_cnt_max[user_id]:
            user_cnt_max[user_id] = locs[loc]






path_stat = "".join((home, "/data/yelp/stat/user_loc_stat.txt"))
f_stat = open(path_stat, "w")
for user_id in user_loc_cnt.keys():
    if user_cnt_max[user_id] >= 5 and user_prob_max[user_id] > 0.5 and len(user_loc_cnt[user_id]) > 1:
        f_stat.write(user_id)
        f_stat.write("\t")
        locs = user_loc_cnt[user_id]
        for loc in locs.keys():
            f_stat.write(loc)
            f_stat.write("\t")
            f_stat.write(str(locs[loc]))
            f_stat.write("\t")
        f_stat.write("\n")


