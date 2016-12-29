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
#include <atomic>

namespace entity2vec {

    class controller {
    private:

        clock_t start;
    public:
        std::shared_ptr<args> args_;
        std::shared_ptr<data> data_;
        std::shared_ptr<matrix> input_;
        std::shared_ptr<matrix> output_;
        std::shared_ptr<model> model_;

        void trainThread(uint32_t threadId);

        void train(std::shared_ptr<args> args);
        void skipgram(model& model, real lr, const std::vector<int64_t>& line, const std::vector<int64_t>& labels, const std::vector<int64_t>& tags);
        void printInfo(real progress, real loss);
        void printWords(std::string word, uint32_t k, uint32_t type);//for type 0:word 1:prod 2:tag

        void saveModel(std::string name);
        void saveModel(std::ostream &ofs_word, std::ostream &ofs_prod, std::ostream &ofs_tag);
        void loadModel(std::istream& in);
        void loadModel(const std::string& filename);

        void saveVectors(const std::string &name);

        void populate_pretraining();


    };
}


#endif //TRAIN_CONTROLLER_H
