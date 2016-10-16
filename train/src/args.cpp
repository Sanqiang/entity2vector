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
        bucket = 2000000;
        minn = 3;
        maxn = 6;
        thread = 1;
        lrUpdateRate = 100;
        t = 1e-4;
        label = "__label__";
        verbose = 2;
        pretrainedVectors = "";
    }

}