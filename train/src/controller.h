//
// Created by Sanqiang Zhao on 10/15/16.
//

#ifndef TRAIN_CONTROLLER_H
#define TRAIN_CONTROLLER_H

#include <vector>
#include <iostream>
#include "real.h"
#include "matrix.h"
#include "model.h"
#include "args.h"
#include "data.h"

namespace entity2vec {

    class controller {
    private:
        std::shared_ptr<args> args_;
        std::shared_ptr<data> dict_;
        std::shared_ptr<matrix> input_;
        std::shared_ptr<matrix> output_;
        std::shared_ptr<model> model_;
        std::atomic<int64_t> tokenCount;
    public:
        void skipgram(model& model, real lr, const std::vector<int32_t>& line);
    };
}


#endif //TRAIN_CONTROLLER_H
