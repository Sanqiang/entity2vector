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

    model::model(std::shared_ptr<matrix> wi, std::shared_ptr<matrix> wo, std::shared_ptr<args> args, std::shared_ptr<data> data, uint32_t seed):
            hidden_(args->dim), output_(wo->m_), grad_(args->dim), rng(seed) {
        wi_ = wi;
        wo_ = wo;
        args_ = args;
        isz_ = wi->m_;
        osz_ = wo->m_;
        hsz_ = args->dim;
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

    real model::binaryLogistic(int64_t target, bool label, real lr) {
        real a = wo_->dotRow(hidden_, target);
        if(isnan(a)){
            printf("%d:%f \n", target, a);
            a = 1000;
        }
        real score = util::sigmoid(a);
        real alpha = lr * (real(label) - score);
        grad_.addRow(*wo_, target, alpha);
        wo_->addRow(hidden_, target, alpha);
        if (label) {
            return -util::log(score);
        } else {
            return -util::log(1.0 - score);
        }
    }

    real model::negativeSampling(int64_t input, int64_t target, real lr) {
        real loss = 0.0;
        grad_.zero();
        for (uint32_t n = 0; n <= args_->neg; n++) {
            if (n == 0) {
                loss += binaryLogistic(target, true, lr);
            } else {
                int64_t neg_target = getNegative(input, target);
                loss += binaryLogistic(neg_target, false, lr);
            }
        }
        return loss;
    }

    uint32_t model::getNegative(int64_t input, int64_t target) {
        int64_t negative = -1;
        if(checkIndexType(input) == 0 && checkIndexType(target) == 0){ //word-word is word
            do {
                negative = word_negatives[negpos_word % word_negatives.size()];
                negpos_word = (negpos_word + 1) % word_negatives.size();
            } while (target == negative);
        }else if(checkIndexType(input) == 0 && checkIndexType(target) == 1){ //
            do {
                negative = prod_negatives[negpos_prod % prod_negatives.size()];
                negpos_prod = (negpos_prod + 1) % prod_negatives.size();
                if(!data_->checkCorPair(input, negative - data_->word_size_, 1)){
                    break;
                }
            } while (1);
        }else if(checkIndexType(input) == 1 && checkIndexType(target) == 0){
            do {
                negative = word_negatives[negpos_word % word_negatives.size()];
                negpos_word = (negpos_word + 1) % word_negatives.size();
                if(!data_->checkCorPair(negative, input - data_->word_size_, 1)){
                    break;
                }
            } while (1);
        }else if(checkIndexType(input) == 0 && checkIndexType(target) == 2){
            do {
                negative = tag_negatives[negpos_tag % tag_negatives.size()];
                negpos_tag = (negpos_tag + 1) % tag_negatives.size();
                if(!data_->checkCorPair(input, negative - data_->word_size_ - data_->prod_size_, 2)){
                    break;
                }
            } while (1);
        }else if(checkIndexType(input) == 2 && checkIndexType(target) == 0){
            do {
                negative = word_negatives[negpos_word % word_negatives.size()];
                negpos_word = (negpos_word + 1) % word_negatives.size();
                if(!data_->checkCorPair(negative, input - data_->word_size_ - data_->prod_size_, 2)){
                    break;
                }
            } while (1);
        }else if(checkIndexType(input) == 1 && checkIndexType(target) == 2){
            do {
                negative = tag_negatives[negpos_tag % tag_negatives.size()];
                negpos_tag = (negpos_tag + 1) % tag_negatives.size();
                if(!data_->checkCorPair(negative  - data_->word_size_ - data_->prod_size_, input- data_->word_size_, 3)){
                    break;
                }
            } while (1);
        }else if(checkIndexType(input) == 2 && checkIndexType(target) == 1){
            do {
                negative = prod_negatives[negpos_prod % prod_negatives.size()];
                negpos_prod = (negpos_prod + 1) % prod_negatives.size();
                if(!data_->checkCorPair(input  - data_->word_size_ - data_->prod_size_, negative- data_->word_size_, 3)){
                    break;
                }
            } while (1);
        }

        return negative;
    }

    void model::computeHidden(int64_t input, vector &hidden) {
        hidden.zero();
        hidden.addRow(*wi_, input);
    }

    void model::initTableNegatives() {
        const std::vector<uint32_t> counts = data_->getWordCounts();
        real z = 0.0;
        for (size_t i = 0; i < counts.size(); i++) {
            z += pow(counts[i], 0.5);
        }
        for (size_t i = 0; i < counts.size(); i++) {
            real c = pow(counts[i], 0.5);
            for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                word_negatives.push_back(i);
            }
        }
        std::shuffle(word_negatives.begin(), word_negatives.end(), rng);

        if(args_->prod_flag){
            const std::vector<uint32_t> counts_prod = data_->getProdCounts();
            real z_prod = 0.0;
            for (size_t i = 0; i < counts_prod.size(); i++) {
                z_prod += pow(counts_prod[i], 0.5);
            }
            for (size_t i = 0; i < counts_prod.size(); i++) {
                real c = pow(counts_prod[i], 0.5);
                for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                    prod_negatives.push_back(i);
                }
            }
            std::shuffle(prod_negatives.begin(), prod_negatives.end(), rng);

            const std::vector<uint32_t> counts_tag = data_->getTagCounts();
            real z_tag = 0.0;
            for (size_t i = 0; i < counts_tag.size(); i++) {
                z_tag += pow(counts_tag[i], 0.5);
            }
            for (size_t i = 0; i < counts_tag.size(); i++) {
                real c = pow(counts_tag[i], 0.5);
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

    void model::update(int64_t input, int64_t target, real lr) {
        computeHidden(input, hidden_);
        loss_ += negativeSampling(input, target, lr);
        nexamples_ += 1;
        wi_->addRow(grad_, input, 1.0);
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

    uint8_t model::checkIndexType(int64_t index) {
        if (index < n_words_){
            return 0;
        }else if(index < n_words_+n_prods_){
            return 1;
        }else if(index < n_words_+n_prods_+n_tags_){
            return 2;
        }
    }
}