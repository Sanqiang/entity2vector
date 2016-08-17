
from gensim.models.word2vec import Word2Vec
if False:
    idx2prod = []
    prod_file = "/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod_aword/prod.txt"
    f_prod = open(prod_file, "r")
    for line in f_prod:
        idx2prod.append(line)

    model = Word2Vec.load_word2vec_format("/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod_aword/output/model_29")
    prods = model.most_similar(idx2prod[1])

    print(prods)

if True:
    idx2user = []
    user_file = "/Users/zhaosanqiang916/git/entity2vector/yelp_rest_user_aword/user.txt"
    f_user = open(user_file, "r")
    for line in f_user:
        idx2user.append(line)

    model = Word2Vec.load_word2vec_format("/Users/zhaosanqiang916/git/entity2vector/yelp_rest_user_aword/output/model_3")
    users = model.most_similar(idx2user[0][0:-1])

    print(users)

if False:
    import json
    data_file = "/Users/zhaosanqiang916/data/yelp/review_rest.json"
    f = open(data_file, "r")
    for line in f:
        obj = json.loads(line)
        title = " "
        text = obj["text"]
        user = obj["user_id"]
        prod = obj["business_id"]
        rating = obj["stars"]

        if user == "":
            print(text)
