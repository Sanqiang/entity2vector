#prepare the data set

import json

if False: #check category list
    f = open("/home/sanqiang/data/yelp/business.json","r")
    categories_set = set()
    for line in f:
        obj = json.loads(line)
        business_id = obj["business_id"]
        categories = obj["categories"]
        for category in categories:
            categories_set.add(category)

    for category in categories_set:
        print(category)

if True: #only generate Restaurants
    f = open("/home/sanqiang/data/yelp/business.json","r")
    interested_business_id = set()

    for line in f:
        obj = json.loads(line)
        business_id = obj["business_id"]
        categories = obj["categories"]
        if "Restaurants" in categories:
            interested_business_id.add(business_id)

    f = open("/home/sanqiang/data/yelp/NV.json","r")
    fu = open("/home/sanqiang/data/yelp/NVu2.json","w")
    nline = ""
    ncount = 0
    for line in f:
        obj = json.loads(line)
        business_id = obj["business_id"]

        if business_id in interested_business_id:
            nline = "".join((nline,line))
        ncount += 1
        if ncount >= 10000:
            fu.write(nline)
            print(ncount)
            ncount = 0
            nline = ""
    fu.write(nline)
    fu.close()

