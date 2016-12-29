//
// Created by Sanqiang Zhao on 10/10/16.
//

#include "model.h"
#include "util.h"
#include <memory>
#include <algorithm>
#include <fstream>
#include "data.h"
#include <unordered_set>

namespace entity2vec {

    model::model(std::shared_ptr<matrix> wi, std::shared_ptr<matrix> wo,
                 std::shared_ptr<matrix> pi, std::shared_ptr<matrix> po, std::shared_ptr<matrix> w2p,
                 std::shared_ptr<matrix> ti, std::shared_ptr<matrix> to, std::shared_ptr<matrix> w2t,
                 std::shared_ptr<args> args, std::shared_ptr<data> data, uint32_t seed): rng(seed) {
        wi_ = wi;
        wo_ = wo;
        pi_ = pi;
        po_ = po;
        ti_ = ti;
        to_ = to;
        w2p_ = w2p;
        w2t_ = w2t;
        args_ = args;
        negpos_word = 0;
        negpos_prod = 0;
        negpos_tag = 0;
        loss_ = 0.0;
        nexamples_ = 1;
        data_ = data;
        n_words_ = data_->word_size_;
        n_prods_ = data_->prod_size_;
        n_tags_ = data_->tag_size_;
    }

    real model::binaryLogistic(int64_t input, int64_t target, bool label, real lr, train_mode mode) {
        real f = 0, g = 0, score = 0;

        if(mode == train_mode::word2prod) {
            // word is target prod is input
            vector temp(args_->dim_p);
            temp.zero();
            for (int64_t p_i = 0; p_i < args_->dim_p; ++p_i) {
                for (int64_t w_i = 0; w_i < args_->dim_w; ++w_i) {
                    temp.incrementData(wi_->getValue(target, w_i) * w2p_->getValue(w_i, p_i), p_i);
                }
            }
            for (int64_t p_i = 0; p_i < args_->dim_p; ++p_i) {
                f += temp.getValue(p_i) * pi_->getValue(input, p_i);
            }

            if (f > MAX_SIGMOID) {
                g = (real(label) - 1) * lr;
            } else if (f < -MAX_SIGMOID) {
                g = (real(label) - 0) * lr;
            } else {
                score = util::exp(f);
                g = (real(label) - score) * lr;
            }



            for (int64_t p_i = 0; p_i < args_->dim_p; ++p_i) {
                neu_vector->incrementData(g * temp.getValue(p_i), p_i);
            }

            for (int64_t p_i = 0; p_i < args_->dim_p; ++p_i) {
                for (int64_t w_i = 0; w_i < args_->dim_w; ++w_i) {
                    neu_matrix->incrementData(wi_->getValue(target, w_i) * pi_->getValue(input, p_i), w_i, p_i);
                }
            }

        }else if(mode == train_mode::word2word){
            for (int64_t w_i = 0; w_i < args_->dim_w; ++w_i) {
                f += wi_->getValue(input, w_i) * wi_->getValue(target, w_i);
            }

            if (f > MAX_SIGMOID) {
                g = (real(label) - 1) * lr;
            } else if (f < -MAX_SIGMOID) {
                g = (real(label) - 0) * lr;
            } else {
                score = util::exp(f);
                g = (real(label) - score) * lr;
            }

            for (int64_t w_i = 0; w_i < args_->dim_w; ++w_i) {
                neu_vector->incrementData(g * wi_->getValue(target, w_i), w_i);
            }
        }

        if (label) {
            return util::log(score);
        } else {
            return util::log(1.0 - score);
        }
    }

    real model::negativeSampling(int64_t input, int64_t target, real lr, train_mode mode) {
        real loss = 0.0;
        for (uint32_t n = 0; n <= args_->neg; n++) {
            if (n == 0) {
                loss += binaryLogistic(input, target, true, lr, mode);
            } else {
                int64_t neg_target = getNegative(input, target, mode);
                if (neg_target == -1)
                    return 0;
                loss += binaryLogistic(input, neg_target, false, lr, mode);
            }
        }
        return loss;
    }

    int64_t model::getNegative(int64_t input, int64_t target, train_mode mode) {
        int64_t negative = -1;
        int64_t cnt = 0;

        if(mode == train_mode::word2word){
            do {
                negative = word_negatives[negpos_word % word_negatives.size()];
                negpos_word = (negpos_word + 1) % word_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
            } while (target == negative);
        } else if(mode == train_mode::word2prod){
            do {
                negative = prod_negatives[negpos_prod % prod_negatives.size()];
                negpos_prod = (negpos_prod + 1) % prod_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
                if(!data_->checkCorPair(input, negative, entity_pair_type::word_prod)){
                    break;
                }
            } while (1);
        }else if(mode == train_mode::word2tag){
            do {
                negative = tag_negatives[negpos_tag % tag_negatives.size()];
                negpos_tag = (negpos_tag + 1) % tag_negatives.size();
                ++cnt;
                if(cnt > args_->neg_trial) return -1;
                if(!data_->checkCorPair(input, negative, entity_pair_type::word_tag)){
                    break;
                }
            } while (1);
        }
        return negative;
    }

    void model::initTableNegatives() {
        const std::vector<uint32_t> counts = data_->getWordCounts();
        real z = 0.0;
        for (size_t i = 0; i < counts.size(); i++) {
            z += pow(counts[i], 0.75);
        }
        for (size_t i = 0; i < counts.size(); i++) {
            real c = pow(counts[i], 0.75);
            for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                word_negatives.push_back(i);
            }
        }
        std::shuffle(word_negatives.begin(), word_negatives.end(), rng);

        if(args_->prod_flag){
            const std::vector<uint32_t> counts_prod = data_->getProdCounts();
            real z_prod = 0.0;
            for (size_t i = 0; i < counts_prod.size(); i++) {
                z_prod += pow(counts_prod[i], 0.75);
            }
            for (size_t i = 0; i < counts_prod.size(); i++) {
                real c = pow(counts_prod[i], 0.75);
                for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                    prod_negatives.push_back(i);
                }
            }
            std::shuffle(prod_negatives.begin(), prod_negatives.end(), rng);

            const std::vector<uint32_t> counts_tag = data_->getTagCounts();
            real z_tag = 0.0;
            for (size_t i = 0; i < counts_tag.size(); i++) {
                z_tag += pow(counts_tag[i], 0.75);
            }
            for (size_t i = 0; i < counts_tag.size(); i++) {
                real c = pow(counts_tag[i], 0.75);
                for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                    tag_negatives.push_back(i);
                }
            }
            std::shuffle(tag_negatives.begin(), tag_negatives.end(), rng);
        }
    }

    void model::initWordNegSampling() {
        initTableNegatives();
    }

    void model::update(int64_t input, int64_t target, real lr, train_mode mode) {
        if(mode == train_mode::word2word){
            neu_vector = new vector(args_->dim_w);
            neu_vector->zero();
            loss_ += negativeSampling(input, target, lr, mode);

            nexamples_ += 1;
            wi_->addRow(* neu_vector, input, 1.0);
        }else if(mode == train_mode::word2prod){
            neu_vector = new vector(args_->dim_p);
            neu_vector->zero();
            neu_matrix = new matrix(args_->dim_w, args_->dim_p);
            neu_matrix->zero();
            loss_ += negativeSampling(input, target, lr, mode);

            nexamples_ += 1;
            pi_->addRow(* neu_vector, input, 1.0);
            w2p_->addMatrix(* neu_matrix, 1.0);

        }

    }

    real model::getLoss() const {
        return loss_ / nexamples_;
    }

    void model::load(std::istream &in) {
        for (int32_t i = 0; i < NEGATIVE_TABLE_SIZE; ++i) {
            int32_t temp;
            in.read((char*) &temp, sizeof(int32_t));
            word_negatives.push_back(temp);
        }
    }

    void model::save(std::ostream &out) {
        for (int32_t i = 0; i < NEGATIVE_TABLE_SIZE; ++i) {
            out.write((char*) &(word_negatives[i]), sizeof(int32_t));
        }
    }
}