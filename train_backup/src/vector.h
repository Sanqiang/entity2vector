//
// Created by Sanqiang Zhao on 10/15/16.
//

#ifndef TRAIN_VECTOR_H
#define TRAIN_VECTOR_H

#include <iostream>
#include "real.h"

namespace entity2vec {
    class matrix;

    class vector {
    public:
        uint32_t m_;
        real* data_;
        explicit vector(uint32_t m);
        explicit vector(uint32_t m, real *arr);
        void setData(real *arr);
        void setData(real val, int64_t i);
        uint32_t size();
        void zero();
        void addRow(const matrix& A, int64_t i);
        void addRow(const matrix& A, int64_t i, real a);
        void mul(real a);
        void mul(const matrix& A, const matrix& vec);
        void normalize();

    };
}


#endif //TRAIN_VECTOR_H
