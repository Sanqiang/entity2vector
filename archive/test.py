
from gensim.models.word2vec import Word2Vec
if False:
    idx2prod = []
    prod_file = "/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod_aword/prod.txt"
    f_prod = open(prod_file, "r")
    for line in f_prod:
        idx2prod.append(line)
    idx = 1;
    model = Word2Vec.load_word2vec_format("/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod_aword/output/model_27")
    prods = model.most_similar(idx2prod[idx][0:-1], topn=len(idx2prod))
    print(idx2prod[idx])
    for i in range(10):
        print(prods[i])
    for i in range(10):
        print(prods[i*-1])

if False:
    idx2user = []
    user_file = "/Users/zhaosanqiang916/git/entity2vector/yelp_rest_user_aword/user.txt"
    f_user = open(user_file, "r")
    for line in f_user:
        idx2user.append(line)
    idx = 1;
    model = Word2Vec.load_word2vec_format("/Users/zhaosanqiang916/git/entity2vector/yelp_rest_user_aword/output/model_45")
    users = model.most_similar(idx2user[idx][0:-1], topn=len(idx2user))
    print(idx2user[idx])
    for i in range(10):
        print(users[i])
    for i in range(10):
        print(users[i*-1])

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

        if user == "BpLD2_SHWh3TV8xxAwEeeA":
            print(text)

if True:
    from scipy.sparse import lil_matrix,dok_matrix
    sm = dok_matrix((3,3))
    sm[1,1] = 1
    sm[0,1] = 2
    for i,j in sm.keys():
        print(i, j)
    print(sm)