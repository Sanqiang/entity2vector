//
// Created by Sanqiang Zhao on 10/10/16.
//

#ifndef TRAIN_MODEL_H
#define TRAIN_MODEL_H

#include "real.h"
#include "matrix.h"
#include "args.h"
#include <random>
#include <iostream>
#include <vector>
#include <memory>
#include "data.h"

namespace entity2vec {
    class model {
        std::shared_ptr<matrix> wi_;
        std::shared_ptr<matrix> wo_;
        std::shared_ptr<args> args_;
        std::shared_ptr<data> data_;

        vector hidden_;
        vector output_;
        vector grad_;
        uint32_t hsz_;
        uint32_t isz_;
        uint32_t osz_;
        uint32_t n_words_;
        uint32_t n_prods_;
        uint32_t n_tags_;
        real loss_;
        uint32_t nexamples_;

        std::vector<uint32_t> word_negatives;
        std::vector<uint32_t> prod_negatives;
        std::vector<uint32_t> tag_negatives;
        size_t negpos_word, negpos_prod, negpos_tag;
    public:
        static const int64_t NEGATIVE_TABLE_SIZE = 10000000;

        model(std::shared_ptr<matrix> wi, std::shared_ptr<matrix> wo, std::shared_ptr<args> args, std::shared_ptr<data> data, uint32_t seed);

        void save(std::ostream& out);
        void load(std::istream& in);

        real binaryLogistic(int64_t target, bool label, real lr);
        real negativeSampling(int64_t input, int64_t target, real lr);
        real getLoss() const;
        int64_t getNegative(int64_t input, int64_t target);

        void computeHidden(int64_t input, vector& hidden);
        void update(int64_t input, int64_t target, real lr);
        void initTableNegatives();
        void initWordNegSampling();

        uint8_t checkIndexType(int64_t index); //0:word 1:prod 2:tag

        std::minstd_rand rng;
    };
}


#endif //TRAIN_MODEL_H
