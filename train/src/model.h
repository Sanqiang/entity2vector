//
// Created by Sanqiang Zhao on 10/10/16.
//

#ifndef TRAIN_MODEL_H
#define TRAIN_MODEL_H

#include "real.h"
#include "matrix.h"
#include "args.h"
#include <random>
#include <iostream>
#include <vector>
#include <memory>
#include "data.h"
#include <unordered_set>

namespace entity2vec {
    enum class train_mode : uint8_t {word2word=0, word2prod=1,word2tag=2};

    class model {
        std::shared_ptr<matrix> wi_, wo_, pi_, po_, ti_, to_, w2p_, w2t_;
        std::shared_ptr<args> args_;
        std::shared_ptr<data> data_;

        uint32_t n_words_;
        uint32_t n_prods_;
        uint32_t n_tags_;
        real loss_;
        uint32_t nexamples_;

        std::vector<uint32_t> word_negatives;
        std::vector<uint32_t> prod_negatives;
        std::vector<uint32_t> tag_negatives;
        size_t negpos_word, negpos_prod, negpos_tag;

        vector * neu_vector;
        matrix * neu_matrix;
    public:
        static const int64_t NEGATIVE_TABLE_SIZE = 10000000;

        model(std::shared_ptr<matrix> wi, std::shared_ptr<matrix> wo,
              std::shared_ptr<matrix> pi, std::shared_ptr<matrix> po, std::shared_ptr<matrix> w2p,
              std::shared_ptr<matrix> ti, std::shared_ptr<matrix> to, std::shared_ptr<matrix> w2t,
              std::shared_ptr<args> args, std::shared_ptr<data> data, uint32_t seed);

        void save(std::ostream& out);
        void load(std::istream& in);

        real binaryLogistic(int64_t input, int64_t target, bool label, real lr, train_mode mode);
        real negativeSampling(int64_t input, int64_t target, real lr, train_mode mode);
        real getLoss() const;
        int64_t getNegative(int64_t input, int64_t target, train_mode mode);

        void update(int64_t input, int64_t target, real lr, train_mode mode);
        void initTableNegatives();
        void initWordNegSampling();

        std::minstd_rand rng;
    };
}


#endif //TRAIN_MODEL_H
