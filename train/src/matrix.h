//
// Created by Sanqiang Zhao on 10/11/16.
//

#ifndef TRAIN_MATRIX_H
#define TRAIN_MATRIX_H

#include <iostream>
#include "real.h"
#include "vector.h"

namespace entity2vec {
    class matrix {
    public:
        real *data_;
        int64_t m_;
        int64_t n_;

        explicit matrix();

        explicit matrix(int64_t m, int64_t n);

        void uniform(real a);
        real dotRow(const vector& vec, int64_t i);
        void addRow(const vector& vec, int64_t i, real a);

    };
}


#endif //TRAIN_MATRIX_H
