//
// Created by Sanqiang Zhao on 10/15/16.
//

#include "vector.h"
#include "matrix.h"

namespace entity2vec {
    vector::vector(int64_t m) {
        m_ = m;
        data_ = new real[m];
    }

    int64_t vector::size() {
        return m_;
    }

    void vector::zero() {
        for (int64_t i = 0; i < m_; i++) {
            data_[i] = 0.0;
        }
    }

    void vector::addRow(const matrix &A, int64_t i) {
        for (int64_t j = 0; j < A.n_; j++) {
            data_[j] += A.data_[i * A.n_ + j];
        }
    }

    void vector::addRow(const matrix &A, int64_t i, real a) {
        for (int64_t j = 0; j < A.n_; j++) {
            data_[j] += a * A.data_[i * A.n_ + j];
        }
    }

    void vector::mul(real a) {
        for (int64_t i = 0; i < m_; i++) {
            data_[i] *= a;
        }
    }

    void vector::mul(const matrix &A, const matrix &vec) {
        for (int64_t i = 0; i < m_; i++) {
            data_[i] = 0.0;
            for (int64_t j = 0; j < A.n_; j++) {
                data_[i] += A.data_[i * A.n_ + j] * vec.data_[j];
            }
        }
    }
}
