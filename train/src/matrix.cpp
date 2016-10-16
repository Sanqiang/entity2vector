//
// Created by Sanqiang Zhao on 10/11/16.
//

#include "matrix.h"
#include <random>

namespace entity2vec{
    matrix::matrix() {
        m_ = 0;
        n_ = 0;
        data_ = nullptr;
    }

    matrix::matrix(int64_t m, int64_t n) {
        m_ = m;
        n_ = n;
        data_ = new real[m * n];
    }

    void matrix::uniform(real a) {
        std::minstd_rand rng(1);
        std::uniform_real_distribution<> uniform(-a, a);
        for (int64_t i = 0; i < (m_ * n_); i++) {
            data_[i] = uniform(rng);
        }
    }

    real matrix::dotRow(const vector &vec, int64_t i) {
        real d = 0.0;
        for (int64_t j = 0; j < n_; j++) {
            d += data_[i * n_ + j] * vec.data_[j];
        }
        return d;
    }

    void matrix::addRow(const vector &vec, int64_t i, real a) {
        for (int64_t j = 0; j < n_; j++) {
            data_[i * n_ + j] += a * vec.data_[j];
        }
    }
}