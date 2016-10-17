//
// Created by Sanqiang Zhao on 10/10/16.
//

#include "model.h"
#include "util.h"

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
        int32_t negative;
        do {
            negative = negatives[negpos];
            negpos = (negpos + 1) % negatives.size();
        } while (target == negative);
        return negative;
    }

    void model::computeHidden(const std::vector<uint32_t> &input, vector &hidden) {
        hidden.zero();
        for (auto it = input.cbegin(); it != input.cend(); ++it) {
            hidden.addRow(*wi_, *it);
        }
        hidden.mul(1.0 / input.size());
    }

    void model::initTableNegatives(const std::vector<uint64_t> &counts) {
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

    void model::setTargetCounts(const std::vector<uint64_t> &counts) {
        initTableNegatives(counts);
    }

}