from w2v_base import  W2V_base
from gensim.models import LdaModel
import json

class W2V_cpp2(W2V_base):
    def __init__(self,n_topic, path, folder):
        self.n_topic = n_topic
        W2V_base.__init__(self, path, folder)

        #process dict
        for prod_id in self.idx2prod.keys():
            prod = self.idx2prod[prod_id]
            n_prod_id = prod_id - len(self.word_count) - 1
            del self.idx2prod[prod_id]
            self.idx2prod[n_prod_id] = prod
            self.prod2idx[prod] = n_prod_id

        for user_id in self.idx2user.keys():
            user = self.idx2user[user_id]
            n_user_id = user_id - len(self.word_count) - len(self.prod2idx) - 1
            del self.idx2user[user_id]
            self.idx2user[n_user_id] = user
            self.user2idx[user] = n_user_id

    def train(self):
        data = []

        doc_id = 0
        for obj in self.data:
            doc_id += 1
            doc = []
            obj_sents = obj["text_data"]
            for obj_sent in obj_sents:
                for pair in obj_sent:
                    if pair[0] >= 0:
                        doc.append((pair[0], doc_id))
            data.append(doc)

        self.ldamodel = LdaModel(corpus=data, id2word=self.idx2word, num_topics=self.n_topic)
        topic = self.ldamodel.print_topic(10)
        self.ldamodel.save("/lda/model_100")
        print(topic)

if __name__ == '__main__':
    lda = W2V_cpp2(100, "/Users/zhaosanqiang916/data/yelp/review_rest.json", "yelp_rest_prod")
    print("init")
    lda.train()
    print("done")