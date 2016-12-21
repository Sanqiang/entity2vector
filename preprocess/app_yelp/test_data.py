import json
import os
import mysql.connector
from bs4 import BeautifulSoup

from util.text_process import TextProcess

home = os.environ["HOME"]


if False:
    path = "".join((home, "/data/yelp/review.json"))

    cnx = mysql.connector.connect(user='root', database='data', password='1')
    cur = cnx.cursor(buffered=True)
    insert_stat = (
      "INSERT INTO review (review_id, user_id, business_id, text, stars) "
      "VALUES (%s, %s, %s, %s, %s)")

    f = open(path, "r")
    for line in f:
        obj = json.loads(line)
        text = str(obj["text"])
        # text = TextProcess.process(text)
        user_id = str(obj["user_id"])
        business_id = str(obj["business_id"])
        review_id = str(obj["review_id"])
        stars = str(obj["stars"])

        cur.execute(insert_stat, (review_id, user_id, business_id, text, stars))

    cnx.commit()
    cnx.close()

if True:
    path = "".join((home, "/data/yelp/review.json"))


