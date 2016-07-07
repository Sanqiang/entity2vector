from w2v_pos import W2V_c
import sys

model = W2V_c("/home/sanqiang/Documents/data/yelp/yelp_academic_dataset_review.json", "yelp_ny_pos")
model.train(sys.maxsize)
