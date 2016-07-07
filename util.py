#combine nyc yelp
from bs4 import BeautifulSoup
import chardet
import json
from collections import defaultdict

#get state based data for yelp challenge
if True:
    target_state = "AZ"

    path_review = "/home/sanqiang/Documents/data/yelp/yelp_academic_dataset_review.json"
    path_bus = "/home/sanqiang/Documents/data/yelp/yelp_academic_dataset_business.json"

    busin2state = {}
    state2cnt = defaultdict(int)
    f_bus = open(path_bus, "r")
    for line in f_bus.readlines():
        obj = json.loads(line)
        state = obj["state"]
        business_id = obj["business_id"]
        busin2state[business_id] = state
        state2cnt[state] += 1

    for state, cnt in state2cnt.items():
        print(state, cnt)

    f_output = open("".join(("/home/sanqiang/Documents/data/yelp/", target_state,".json")), mode="w")
    f_review = open(path_review, "r")
    for line in f_review.readlines():
        obj = json.loads(line)
        business_id = obj["business_id"]
        if busin2state[business_id] in target_state:
            f_output.write(line)


#combine ny yelp
if False:
    path1 = "/home/sanqiang/Documents/data/yelp data/NewYork/reviews_NewYork.txt"
    path2 = "/home/sanqiang/Documents/data/yelp data/NewYork/ny.txt"
    f1 = open(path1, "r",encoding='cp1252', errors='ignore')
    f2 = open(path2, "a",encoding='cp1252', errors='ignore')
    f2.writelines(f1.readlines())