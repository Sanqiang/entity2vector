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
        uint32_t dim_w, dim_p, dim_t;
        uint32_t ws;
        uint32_t epoch;
        uint32_t minCount;
        uint32_t neg;
        uint32_t neg_trial;
        uint32_t wordNgrams;
        //loss_name loss;
        //model_name model;
        //uint32_t bucket;
        uint32_t minn;
        uint32_t maxn;
        uint32_t thread;
        uint8_t memory_mode; //0 leave data in disk; 1 leave data in memory
        double t;
        std::string label;
        uint8_t verbose;
        std::string pretrainedVectors;

        uint8_t prod_flag;
        uint8_t tag_flag;
        uint8_t mode_flag; //0 means word2vec; 1 means prod2vec; 2 means proc&tag/vec
        uint8_t neg_flag; //0 means hashtable; 1 means table
        uint8_t pretraining_flag;
        uint8_t load_model_flag;
        std::string load_model;

        args();

        void save(std::ostream& out);
        void load(std::istream& in);
    };
}

#endif //TRAIN_ARGS_H
