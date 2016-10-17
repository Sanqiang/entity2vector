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
        uint64_t m_;
        uint64_t n_;

        explicit matrix();

        explicit matrix(uint64_t m, uint64_t n);

        void uniform(real a);
        void zero();
        real dotRow(const vector& vec, uint64_t i);
        void addRow(const vector& vec, uint64_t i, real a);

    };
}


#endif //TRAIN_MATRIX_H
