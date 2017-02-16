from data import DataProvider
from config import Config
import numpy as np
from scipy.sparse import dok_matrix
import json
import heapq
import time
import datetime
from gensim.models import Word2Vec

conf = Config("prod", "prod", 200)
dp = DataProvider(conf)

model = np.load(conf.path_model_npy + ".npy")
word_embed = model[0]
prod_embed = model[1]
transfer_w = model[2]
transfer_b = model[3]

# prepare prod file
dp.prod2idx = {}
batch = ""
f_prod = open("prod_embed", "w")
f_prod.write(str(len(dp.idx2prod)))
f_prod.write(" ")
f_prod.write(str(300))
for prod_idx in range(len(dp.idx2prod)):
    prod = dp.idx2prod[prod_idx]
    dp.prod2idx[prod] = prod_idx

    line = " ".join([prod] + [str(v) for v in prod_embed[prod_idx,]])
    batch = "\n".join([batch, line])
    if len(batch) > 100000:
        f_prod.write(batch)
        batch = ""
f_prod.write(batch)

# process data
h = []
datas = []
usr2idx = {}
idx2usr = []
with open(conf.path_raw_data, "r") as ins:
    for line in ins:
        obj = json.loads(line)
        business_id = obj["business_id"]
        if business_id not in dp.prod2idx:
            continue

        usr = obj["user_id"]
        if usr not in usr2idx:
            usr2idx[usr] = len(usr2idx)
            idx2usr.append(usr)
        rating = obj["stars"]

        unixReviewTime = time.mktime(datetime.datetime.strptime(obj["date"], "%Y-%m-%d").timetuple()) #int(obj["date"])
        datas.append((usr, business_id, rating, unixReviewTime))
        heapq.heappush(h, unixReviewTime)
split = int(len(h) * 0.8)
datas_train = []
datas_test = []
for entry in datas:
    unixReviewTime = entry[3]
    if unixReviewTime < split:
        datas_train.append(entry)
    else:
        datas_test.append(entry)
del datas

train_mat = dok_matrix((len(idx2usr), len(dp.idx2prod)), dtype=np.float)
for entry in datas_train:
    train_mat[usr2idx[entry[0]], dp.prod2idx[entry[1]]] = entry[2]

test_mat = dok_matrix((len(idx2usr), len(dp.idx2prod)), dtype=np.float)
for entry in datas_test:
    test_mat[usr2idx[entry[0]], dp.prod2idx[entry[1]]] = entry[2]

#populate entry matrix
entity_model = Word2Vec.load_word2vec_format("prod_embed")

rmse = []
for usr_idx, prod_idx in test_mat.keys():
    true_rating = test_mat[usr_idx, prod_idx]

    #predict
    inds = (train_mat[usr_idx, :] > 0).indices
    inds = [dp.idx2prod[ind] for ind in inds]
    list = entity_model.most_similar(dp.idx2prod[prod_idx], topn=7, restrict_vocab=inds)

    preidct_rating = 0
    denom = 1
    for pair in list:
        pair_ent = pair[0]
        prod = pair_ent[0:pair_ent.rindex("_")]
        prod_id = pair_ent[1+pair_ent.rindex("_"):]
        dist = pair[1]
        rating = train_mat[usr_idx, prod_idx]
        preidct_rating += rating * dist
        denom += dist
    preidct_rating /= denom

    rmse.append((true_rating - preidct_rating) ** 2)
print(np.mean(rmse))

