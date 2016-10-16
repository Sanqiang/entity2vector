//
// Created by Sanqiang Zhao on 10/15/16.
//

#include "controller.h"

namespace entity2vec{
    void controller::skipgram(model &model, real lr, const std::vector<int32_t> &line) {
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (int32_t w = 0; w < line.size(); w++) {
            int32_t boundary = uniform(model.rng);
            //const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w]);
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    //model.update(ngrams, line[w + c], lr);
                }
            }
        }
    }
}