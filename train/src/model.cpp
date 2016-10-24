//
// Created by Sanqiang Zhao on 10/10/16.
//

#include "model.h"
#include "util.h"
#include <memory>
#include <algorithm>
#include <fstream>
#include "data.h"

namespace entity2vec {

    model::model(std::shared_ptr<matrix> wi, std::shared_ptr<matrix> wo, std::shared_ptr<args> args, uint32_t seed):
            hidden_(args->dim), output_(wo->m_), grad_(args->dim), rng(seed) {
        wi_ = wi;
        wo_ = wo;
        args_ = args;
        isz_ = wi->m_;
        osz_ = wo->m_;
        hsz_ = args->dim;
        negpos = 0;
        loss_ = 0.0;
        nexamples_ = 1;
    }

    real model::binaryLogistic(uint32_t target, bool label, real lr) {
        real score = util::sigmoid(wo_->dotRow(hidden_, target));
        real alpha = lr * (real(label) - score);
        grad_.addRow(*wo_, target, alpha);
        wo_->addRow(hidden_, target, alpha);
        if (label) {
            return -util::log(score);
        } else {
            return -util::log(1.0 - score);
        }
    }

    real model::negativeSampling(uint32_t target, real lr) {
        real loss = 0.0;
        grad_.zero();
        for (uint32_t n = 0; n <= args_->neg; n++) {
            if (n == 0) {
                loss += binaryLogistic(target, true, lr);
            } else {
                loss += binaryLogistic(getNegative(target), false, lr);
            }
        }
        return loss;
    }

    uint32_t model::getNegative(uint32_t target) {
        uint32_t negative;
        do {
            negative = negatives[negpos];
            negpos = (negpos + 1) % negatives.size();
        } while (target == negative);
        return negative;
    }

    void model::computeHidden(uint32_t input, vector &hidden) {
        hidden.zero();
        hidden.addRow(*wi_, input);
    }

    void model::initTableNegatives(const std::vector<uint32_t> &counts) {
        real z = 0.0;
        for (size_t i = 0; i < counts.size(); i++) {
            z += pow(counts[i], 0.5);
        }
        for (size_t i = 0; i < counts.size(); i++) {
            real c = pow(counts[i], 0.5);
            for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                negatives.push_back(i);
            }
        }
        std::shuffle(negatives.begin(), negatives.end(), rng);
    }

    void model::setTargetCounts(const std::vector<uint32_t> &counts) {
        initTableNegatives(counts);
    }

    void model::update(uint32_t input, uint32_t target, real lr) {
        computeHidden(input, hidden_);
        loss_ += negativeSampling(target, lr);
        nexamples_ += 1;
        wi_->addRow(grad_, input, 1.0);
    }

    real model::getLoss() const {
        return loss_ / nexamples_;
    }
}