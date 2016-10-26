//
// Created by Sanqiang Zhao on 10/11/16.
//

#include "matrix.h"
#include <queue>
#include <vector>
#include <iostream>
#include <random>

namespace entity2vec{
    matrix::matrix() {
        m_ = 0;
        n_ = 0;
        data_ = nullptr;
    }

    matrix::matrix(uint32_t m, uint32_t n) {
        m_ = m;
        n_ = n;
        data_ = new real[m * n];
    }

    void matrix::uniform(real a) {
        std::minstd_rand rng(1);
        std::uniform_real_distribution<> uniform(-a, a);
        for (uint32_t i = 0; i < (m_ * n_); i++) {
            data_[i] = uniform(rng);
        }
    }

    void matrix::zero() {
        for (uint32_t i = 0; i < (m_ * n_); i++) {
            data_[i] = 0.0;
        }
    }

    real matrix::dotRow(const vector &vec, uint32_t i) {
        real d = 0.0;
        for (uint32_t j = 0; j < n_; j++) {
            d += data_[i * n_ + j] * vec.data_[j];
        }
        return d;
    }

    void matrix::addRow(const vector &vec, uint32_t i, real a) {
        for (uint32_t j = 0; j < n_; j++) {
            data_[i * n_ + j] += a * vec.data_[j];
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

    void matrix::setValue(uint32_t m, uint32_t n, real val) {
        data_[m*n_ + n] = val;
    }

    std::vector<std::pair<real, int>> matrix::findSimilarRow(uint32_t i, uint32_t k, uint32_t range_start, uint32_t range_end) {
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
                temp.setData(data_[cand*n_ + di],di);
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