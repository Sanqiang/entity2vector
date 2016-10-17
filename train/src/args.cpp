//
// Created by Sanqiang Zhao on 10/15/16.
//

#include "args.h"

namespace entity2vec {
    args::args() {
        lr = 0.05;
        dim = 100;
        ws = 5;
        epoch = 5;
        minCount = 5;
        neg = 5;
        wordNgrams = 1;
        minn = 0;
        maxn = 6;
        thread = 1;
        lrUpdateRate = 100;
        t = 1e-4;
        label = "__label__";
        verbose = 2;
        pretrainedVectors = "";
        input = "/Users/zhaosanqiang916/data/yelp/review_processed_sample.txt";
    }

}