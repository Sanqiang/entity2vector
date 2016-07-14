from w2v_s import W2V_c
import sys

#model = W2V_c("/home/sanqiang/Documents/data/yelp/yelp_academic_dataset_review.json", "yelp_ny_pos")
model = W2V_c("/home/sanqiang/Documents/data/Amazon_Instant_Video_5.json", "amazon_instant_video")
model.train(sys.maxsize)
