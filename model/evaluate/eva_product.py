from config import Config
from data import DataProvider
from gensim.models.word2vec import Word2Vec
import numpy as np
import os

flag = "Adam"
conf = Config(flag)

if not os.path.exists(conf.path_word_w2c) and not os.path.exists(conf.path_doc_w2c):
    doc_embed = np.load(conf.path_doc_npy + ".npy")[0]
    dp = DataProvider(conf)

    # generate doc embedding file
    f = open(conf.path_doc_w2c,"w")
    f.write(str(len(dp.idx2prod)))
    f.write(" ")
    f.write(str(conf.dim_item))
    f.write("\n")
    idx = 0
    batch = ""
    for word in dp.idx2prod:
        batch = "".join([batch, word])
        batch = "".join([batch, " "])

        for i in range(conf.dim_item):
            batch = "".join([batch, str(doc_embed[idx][i])])
            batch = "".join([batch, " "])

        batch = "".join([batch, "\n"])
        idx += 1
        if len(batch) > 100000:
            f.write(batch)
            batch = ""
    f.write(batch)

    word_embed = np.load(conf.path_word_npy + ".npy")[0]
    dp = DataProvider(conf)

    # generate word embedding file
    f = open(conf.path_word_w2c,"w")
    f.write(str(len(dp.idx2word)))
    f.write(" ")
    f.write(str(conf.dim_word))
    f.write("\n")
    idx = 0
    batch = ""
    for word in dp.idx2word:
        batch = "".join([batch, word])
        batch = "".join([batch, " "])

        for i in range(conf.dim_word):
            batch = "".join([batch, str(word_embed[idx][i])])
            batch = "".join([batch, " "])

        batch = "".join([batch, "\n"])
        idx += 1
        if len(batch) > 100000:
            f.write(batch)
            batch = ""
    f.write(batch)
    print("finish generate")

# describe model
# dp.prod2idx issue
# repopulate prod2idx
dp = DataProvider(conf)
dp.prod2idx = {}
for idx, prod in enumerate(dp.idx2prod):
    dp.prod2idx[prod] = idx
describe_model = np.load(conf.path_model_npy + ".npy")
word_embed = describe_model[0]
prod_embed = describe_model[1]
transfer_w = describe_model[2]
transfer_b = describe_model[3]


# test doc
model = Word2Vec.load_word2vec_format(conf.path_doc_w2c)
print("init")
while True:
        source = input()
        source_id = dp.prod2idx[source]
        source_embed = prod_embed[source_id, :]
        source_embed = np.exp(source_embed)
        source_embed = source_embed / source_embed.sum()

        targets = model.most_similar(source)
        diffs = []
        topic_ids = set()
        for target in targets:
            target = target[0]
            target_id = dp.prod2idx[target]
            target_embed = prod_embed[target_id, :]
            target_embed = np.exp(target_embed)
            target_embed = target_embed / target_embed.sum()

            diff = (source_embed - target_embed) ** 2
            diff = np.argsort(diff)[::-1][:15]
            print(target, diff)
            diffs.append(diff)
            for topic_id in diff:
                topic_ids.add(topic_id)
        print("=======")

        final_topic_ids = []
        for topic_id in topic_ids:
            add_cur_id = True
            for diff in diffs:
                if not topic_id in diff:
                    add_cur_id = False
                    break
            if add_cur_id:
                final_topic_ids.append(topic_id)
        print(final_topic_ids)



