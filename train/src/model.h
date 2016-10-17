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

namespace entity2vec {
    class model {
        std::shared_ptr<matrix> wi_;
        std::shared_ptr<matrix> wo_;
        std::shared_ptr<args> args_;

        vector hidden_;
        vector output_;
        vector grad_;
        uint32_t hsz_;
        uint32_t isz_;
        uint32_t osz_;
        real loss_;
        uint32_t nexamples_;

        std::vector<uint32_t> negatives;
        size_t negpos;

        static const int32_t NEGATIVE_TABLE_SIZE = 10000000;
    public:
        model(std::shared_ptr<matrix> wi, std::shared_ptr<matrix> wo, std::shared_ptr<args> args, uint32_t seed);

        real binaryLogistic(uint32_t target, bool label, real lr);
        real negativeSampling(uint32_t target, real lr);
        real getLoss() const;
        uint32_t getNegative(uint32_t target);

        void computeHidden(uint32_t input, vector& hidden);
        void initTableNegatives(const std::vector<uint32_t>& counts);
        void update(uint32_t input, uint32_t target, real lr);
        void setTargetCounts(const std::vector<uint32_t>& counts);

        std::minstd_rand rng;
    };
}


#endif //TRAIN_MODEL_H
