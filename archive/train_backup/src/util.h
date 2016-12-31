//
// Created by Sanqiang Zhao on 10/10/16.
//

#ifndef TRAIN_UTIL_H
#define TRAIN_UTIL_H

#define SIGMOID_TABLE_SIZE 5120
#define MAX_SIGMOID 8
#define LOG_TABLE_SIZE 5120
#define EXP_TABLE_SIZE 5120

#include "real.h"
#include <iostream>

namespace entity2vec {
    namespace util {
        real log(real x);
        real sigmoid(real x);
        real exp(real x);

        void initTables();
        void initSigmoid();
        void initLog();
        void initExp();
    }
}


#endif //TRAIN_UTIL_H
