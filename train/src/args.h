//
// Created by Sanqiang Zhao on 10/15/16.
//

#ifndef TRAIN_ARGS_H
#define TRAIN_ARGS_H

#include <string>

namespace entity2vec {
    class args {
    public:
        std::string input_data;
        std::string input_data_pattern;
        std::string input_pretrain;
        std::string test;
        std::string output;
        double lr;
        uint32_t lrUpdateRate;
        uint32_t dim;
        uint32_t ws;
        uint32_t epoch;
        uint32_t minCount;
        uint32_t neg;
        uint32_t wordNgrams;
        //loss_name loss;
        //model_name model;
        //uint32_t bucket;
        uint32_t minn;
        uint32_t maxn;
        uint32_t thread;
        double t;
        std::string label;
        uint8_t verbose;
        std::string pretrainedVectors;

        args();

        void save(std::ostream& out);
        void load(std::istream& in);
    };
}

#endif //TRAIN_ARGS_H
