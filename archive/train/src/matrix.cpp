//
// Created by Sanqiang Zhao on 10/11/16.
//

#include "matrix.h"
#include <queue>
#include <vector>
#include <iostream>
#include <random>
#include <stdlib.h>

namespace entity2vec{
    matrix::matrix() {
        m_ = 0;
        n_ = 0;
        data_ = nullptr;
    }

    matrix::matrix(uint64_t m, uint64_t n) {
        m_ = m;
        n_ = n;
        data_ = new real[m * n];
    }

    void matrix::simple_rand() {
        for (uint64_t i = 0; i < (m_ * n_); i++) {
            data_[i] = (rand() % (10+1)) / 10;
        }
    }

    void matrix::uniform(real a) {
        std::minstd_rand rng(1);
        std::uniform_real_distribution<> uniform(-a, a);
        for (uint64_t i = 0; i < (m_ * n_); i++) {
            data_[i] = uniform(rng);
        }
    }

    void matrix::zero() {
        for (uint64_t i = 0; i < (m_ * n_); i++) {
            data_[i] = 0.0;
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

    void matrix::addMatrix(const matrix &mat, real a) {
        for (uint64_t i = 0; i < m_; i++) {
            for (int64_t j = 0; j < n_; j++) {
                data_[i * n_ + j] += a * mat.data_[i * n_ + j];
            }
        }
    }

    void matrix::normalize(int64_t row_idx) {
        real norm = 0.0;
        for (uint64_t i = 0; i < n_; i++) {
            norm += data_[row_idx * n_ + i] * data_[row_idx * n_ + i];
        }
        norm = sqrt(norm);
        for (uint64_t i = 0; i < n_; i++) {
            data_[row_idx * n_ + i] /= norm;
            if ( data_[row_idx * n_ + i] > 100){
                std::cout << "Warning" << data_[row_idx * n_ + i] << std::endl;
            }
        }
    }

    void matrix::save(std::ostream &out) {
        out.write((char*) &m_, sizeof(uint32_t));
        out.write((char*) &n_, sizeof(uint32_t));
        out.write((char*) data_, m_ * n_ * sizeof(real));
    }

    void matrix::load(std::istream &in) {
        in.read((char*) &m_, sizeof(uint32_t));
        in.read((char*) &n_, sizeof(uint32_t));
        delete[] data_;
        data_ = new real[m_ * n_];
        in.read((char*) data_, m_ * n_ * sizeof(real));
    }

    void matrix::setValue(int64_t row_idx, int64_t col_idx, real val) {
        data_[row_idx*n_ + col_idx] = val;
    }

    real matrix::getValue(int64_t row_idx, int64_t col_idx) {
        return data_[row_idx*n_ + col_idx];
    }

    void matrix::incrementData(real val, int64_t row_idx, int64_t col_idx) {
        data_[row_idx*n_ + col_idx] += val;
    }

    std::vector<std::pair<real, int>> matrix::findSimilarRow(int64_t i, uint32_t k, uint32_t range_start, uint32_t range_end) {
        std::priority_queue<std::pair<real, int>> q;

        real *target_arr, *temp_arr;
        target_arr = new real[n_];
        for (uint32_t di = 0; di < n_; ++di) {
            target_arr[di] = data_[i * n_ + di];
        }
        vector target(n_, target_arr), temp(n_);
        target.normalize();

        for (uint32_t cand = range_start; cand <= range_end; ++cand) {
            for (uint32_t di = 0; di < n_; ++di) {
                temp.setValue(data_[cand * n_ + di], di);
            }
            temp.normalize();

            //cosine sim
            real sim = 0;
            for (uint32_t di = 0; di < n_; ++di) {
                sim += temp.data_[di] * target.data_[di];
            }
            q.push(std::pair<real , int>(sim, cand));
        }

        std::vector<std::pair<real, int>> result;
        for (uint32_t i = 0; i < k; ++i) {
            std::pair<double, int> pair = q.top();
            result.push_back(pair);
            q.pop();
        }
        return result;
    }
}