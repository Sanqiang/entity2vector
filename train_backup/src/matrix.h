//
// Created by Sanqiang Zhao on 10/11/16.
//

#ifndef TRAIN_MATRIX_H
#define TRAIN_MATRIX_H

#include <iostream>
#include <vector>
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

        void simple_rand();
        void uniform(real a);
        void zero();
        real dotRow(const vector& vec, int64_t i);
        void addRow(const vector& vec, int64_t i, real a);

        void save(std::ostream& out);
        void load(std::istream& in);

        void setValue(int64_t row_idx, int64_t col_idx, real val);
        real getValue(int64_t row_idx, int64_t col_idx);

        std::vector<std::pair<real, int>> findSimilarRow(int64_t target, uint32_t k, uint32_t range_start, uint32_t range_end);
    };
}


#endif //TRAIN_MATRIX_H
