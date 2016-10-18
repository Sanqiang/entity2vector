//
// Created by Sanqiang Zhao on 10/15/16.
//

#include "controller.h"

#include "data.h"
#include "matrix.h"
#include <fstream>
#include <thread>
#include <iomanip>

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

        input_ = std::make_shared<matrix>(data_->nwords() + data_->nprods(), args_->dim);
        input_->uniform(1.0 / args_->dim);

        output_ = std::make_shared<matrix>(data_->nwords() + data_->nprods(), args_->dim);
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

        std::vector<uint32_t> line;
        std::vector<uint32_t> labels;
        const uint32_t ntokens = data_->nwords();
        uint32_t localTokenCount = 0;
        while (tokenCount < args_->epoch * ntokens){
            real progress = real(tokenCount) / (args_->epoch * ntokens);
            real lr = args_->lr * (1.0 - progress);
            localTokenCount += data_->getLine(ifs, line, labels, model.rng);
            skipgram(model, lr, line);

            if (localTokenCount > args_->lrUpdateRate) {
                tokenCount += localTokenCount;
                localTokenCount = 0;
                if (threadId == 0 && args_->verbose > 1) {
                    printInfo(progress, model.getLoss());
                }
            }
        }
        if (threadId == 0 && args_->verbose > 0) {
            printInfo(1.0, model.getLoss());
            std::cout << std::endl;
        }
        ifs.close();
    }

    void controller::printInfo(real progress, real loss) {
        real t = real(clock() - start) / CLOCKS_PER_SEC;
        real wst = real(tokenCount) / t;
        real lr = args_->lr * (1.0 - progress);
        int eta = int(t / progress * (1 - progress) / args_->thread);
        int etah = eta / 3600;
        int etam = (eta - etah * 3600) / 60;
        std::cout << std::fixed;
        std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
        std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
        std::cout << "  lr: " << std::setprecision(6) << lr;
        std::cout << "  loss: " << std::setprecision(6) << loss;
        std::cout << "  eta: " << etah << "h" << etam << "m ";
        std::cout << std::flush;
    }

    void controller::skipgram(model &model, real lr, const std::vector<uint32_t> &line) {
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (int32_t w = 0; w < line.size(); w++) {
            int32_t boundary = uniform(model.rng);
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    model.update(line[w], line[w + c], lr);
                }
            }
        }
    }

    void controller::saveModel() {
        std::ofstream ofs(args_->output + ".bin", std::ofstream::binary);
        if (!ofs.is_open()) {
            std::cerr << "Model file cannot be opened for saving!" << std::endl;
            exit(EXIT_FAILURE);
        }
        args_->save(ofs);
        data_->save(ofs);
        input_->save(ofs);
        output_->save(ofs);
        ofs.close();
    }

    void controller::loadModel(std::istream &in) {}

    void controller::loadModel(const std::string &filename) {}
}