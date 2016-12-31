import os
from vector_util import vector

home = os.environ["HOME"]
name = "test1"
path_prod_vector = "".join((home, "/data/model/",name,".prod.vec"))
path_tag_vector = "".join((home, "/data/model/",name,".tag.vec"))
path_word_vector = "".join((home, "/data/model/",name,".word.vec"))


path1 = "".join((home, "/data/model/","test",".1.vec"))
path2 = "".join((home, "/data/model/","test",".2.vec"))
vec = vector()
vec.load_word2vec_format(path1)
print(vec)
vec.append_word2vec_format(path2)
print(vec)