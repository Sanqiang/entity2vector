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
        std::shared_ptr<data> data_;
        std::shared_ptr<matrix> input_;
        std::shared_ptr<matrix> output_;
        std::shared_ptr<model> model_;
        std::atomic<uint32_t> tokenCount;

        clock_t start;
    public:
        void trainThread(uint32_t threadId);

        void train(std::shared_ptr<args> args);
        void skipgram(model& model, real lr, const std::vector<uint32_t>& line);
        void printInfo(real progress, real loss);

        void saveModel();
        void loadModel(std::istream& in);
        void loadModel(const std::string& filename);

    };
}


#endif //TRAIN_CONTROLLER_H
