//
// Created by Sanqiang Zhao on 10/15/16.
//

#ifndef TRAIN_ARGS_H
#define TRAIN_ARGS_H

#include <string>

namespace entity2vec {
    class args {
    public:
        std::string input;
        std::string test;
        std::string output;
        double lr;
        int lrUpdateRate;
        int dim;
        int ws;
        int epoch;
        int minCount;
        int neg;
        int wordNgrams;
        //loss_name loss;
        //model_name model;
        int bucket;
        int minn;
        int maxn;
        int thread;
        double t;
        std::string label;
        int verbose;
        std::string pretrainedVectors;

        args();
    };
}

#endif //TRAIN_ARGS_H
