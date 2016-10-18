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
        uint32_t m_;
        uint32_t n_;

        explicit matrix();

        explicit matrix(uint32_t m, uint32_t n);

        void uniform(real a);
        void zero();
        real dotRow(const vector& vec, uint32_t i);
        void addRow(const vector& vec, uint32_t i, real a);

        void save(std::ostream& out);
        void load(std::istream& in);
    };
}


#endif //TRAIN_MATRIX_H
