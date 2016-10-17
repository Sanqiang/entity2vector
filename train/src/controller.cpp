//
// Created by Sanqiang Zhao on 10/15/16.
//

#include "controller.h"

#include "data.h"
#include "matrix.h"
#include <fstream>
#include <thread>

namespace entity2vec{

    void controller::train(std::shared_ptr<args> args) {
        args_ = args;
        data_ = std::make_shared<data>(args_);

        std::ifstream ifs(args_->input);
        if (!ifs.is_open()) {
            std::cerr << "Input file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }

        data_->readFromFile(ifs);
        ifs.close();

        input_ = std::make_shared<matrix>(data_->nwords(), args_->dim);
        input_->uniform(1.0 / args_->dim);

        output_ = std::make_shared<matrix>(data_->nwords(), args_->dim);
        output_->zero();

        start = clock();
        tokenCount = 0;
        std::vector<std::thread> threads;
        for (uint32_t i = 0; i < args_->thread; i++) {
            threads.push_back(std::thread([=]() { trainThread(i); }));
        }
        for (auto it = threads.begin(); it != threads.end(); ++it) {
            it->join();
        }
    }

    void controller::trainThread(uint32_t threadId) {
        std::ifstream ifs(args_->input);

        model model(input_, output_, args_, threadId);
        model.setTargetCounts(data_->getCounts());

        printf("Thread Id: %d", threadId);

    }


    void controller::skipgram(model &model, real lr, const std::vector<uint32_t> &line) {
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