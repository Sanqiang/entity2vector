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
        int32_t hsz_;
        int32_t isz_;
        int32_t osz_;
        real loss_;
        int64_t nexamples_;

        std::vector<int32_t> negatives;
        size_t negpos;

        static const int32_t NEGATIVE_TABLE_SIZE = 10000000;
    public:
        real binaryLogistic(int32_t target, bool label, real lr);
        real negativeSampling(int32_t target, real lr);
        real getLoss() const;
        int32_t getNegative(int32_t target);

        void computeHidden(const std::vector<int32_t>& input, vector& hidden);
        void initTableNegatives(const std::vector<int64_t>& counts);
        void update(const std::vector<int32_t>& input, int32_t target, real lr);

        std::minstd_rand rng;
    };
}


#endif //TRAIN_MODEL_H
