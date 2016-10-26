//
// Created by Sanqiang Zhao on 10/15/16.
//

#include "controller.h"

#include "data.h"
#include "matrix.h"
#include <fstream>
#include <thread>
#include <iomanip>
#include "util.h"
#include <regex>

namespace entity2vec{

    void controller::train(std::shared_ptr<args> args) {
        args_ = args;
        data_ = std::make_shared<data>(args_);

        std::ifstream ifs(args_->input_data);
        if (!ifs.is_open()) {
            std::cerr << "Input file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout<<"start reading file"<<std::endl;
        data_->readFromFile(ifs);
        ifs.close();
        std::cout<<"finish reading file"<<std::endl;
        input_ = std::make_shared<matrix>(data_->nwords() + data_->nprods(), args_->dim);
        input_->uniform(1.0 / args_->dim);

        output_ = std::make_shared<matrix>(data_->nwords() + data_->nprods(), args_->dim);
        output_->zero();

        //std::cout<<"start reading pretraining file"<<std::endl;
        //populate_pretraining();
        //std::cout<<"finish reading pretraining file"<<std::endl;

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
        //std::ifstream ifs(args_->input_data);
        std::string path = args_->input_data_pattern;
        path.replace(path.find("{i}"), std::string("{i}").size(),std::to_string(threadId));
        std::ifstream ifs(path);
        std::cout<<"start trainThread: "<< threadId <<  ":" <<path<<std::endl;

        model model(input_, output_, args_, threadId);
        model.setTargetCounts(data_->getCounts());

        std::vector<uint32_t> line;
        std::vector<uint32_t> labels;
        const uint32_t ntokens = data_->nwords();
        uint32_t localTokenCount = 0;
        uint32_t loop = 0;
        while (tokenCount < args_->epoch * ntokens){
            real progress = real(tokenCount) / (args_->epoch * ntokens);
            real lr = args_->lr * (1.0 - progress);
            localTokenCount += data_->getLine(ifs, line, labels, model.rng);
            skipgram(model, lr, line, labels);

            if (localTokenCount > args_->lrUpdateRate) {
                tokenCount += localTokenCount;
                localTokenCount = 0;
                if (loop++ % 30000 == 0 && threadId == 0 && args_->verbose > 1) {
                    printInfo(progress, model.getLoss());
                    saveModel("test" + std::to_string(threadId));
                }
            }
        }
        if (threadId == 0 && args_->verbose > 0) {
            printInfo(1.0, model.getLoss());
            std::cout << std::endl;
        }
        ifs.close();
    }

    void controller::populate_pretraining() {
        std::ifstream ifs(args_->input_pretrain);
        if (!ifs.is_open()) {
            std::cerr << "Input file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        uint8_t cur_mode = 0; // 0 for words, 1 for vector
        uint32_t cur_vector_idx = 0, cur_word_idx;
        std::string word;
        char c;
        std::streambuf& sb = *ifs.rdbuf();
        while ((c = sb.sbumpc()) != EOF) {
            if(c == '\n'){ break;}
        } //pass the first line
        while ((c = sb.sbumpc()) != EOF) {
            if (c == ' ' || c == '\n') {
                if(!word.empty()){
                    if(cur_mode == 0){
                        cur_word_idx = data_->getWordId(word);
                        cur_vector_idx = 0;
                    }else if(cur_mode == 1){
                        input_->setValue(cur_word_idx, cur_vector_idx++, stod(word));
                    }
                }

                if(c == ' ' || cur_mode == 0){
                    cur_mode = 1;
                }else if(c == '\n'){
                    cur_mode = 0;
                }
                word.clear();
            }else{
                word.push_back(c);
            }
        }
    }

    void controller::printWords(std::string word, uint32_t k) {
        uint32_t i = data_->getWordId(word);
        std::vector<std::pair<real, int>> pairs = input_->findSimilarRow(i, k, 0, data_->nwords()-1);

        std::cout << "" <<word<< " : ";
        for (auto it = pairs.begin(); it != pairs.end(); ++it){
            std::cout << data_->getWord(it->second) << "\t";
        }
        std::cout << std::endl;
    }

    void controller::printInfo(real progress, real loss) {
        real t = real(clock() - start) / CLOCKS_PER_SEC;
        real wst = real(tokenCount) / t;
        real lr = args_->lr * (1.0 - progress);
        int eta = int(t / progress * (1 - progress) / args_->thread);
        int etah = eta / 3600;
        int etam = (eta - etah * 3600) / 60;
        std::cout << std::fixed;
        std::cout << "Progress: " << std::setprecision(1) << 100 * progress << "%";
        std::cout << "words/sec/thread: " << std::setprecision(0) << wst;
        std::cout << "lr: " << std::setprecision(6) << lr;
        std::cout << "loss: " << std::setprecision(6) << loss;
        std::cout << "eta: " << etah << "h" << etam << "m ";
        std::cout << std::endl;
        printWords("steak",10);
        printWords("seafood",10);
        printWords("yummy",10);
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::flush;
    }

    void controller::skipgram(model &model, real lr, const std::vector<uint32_t> &line, const std::vector<uint32_t> &label) {
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (uint32_t w = 0; w < line.size(); w++) {
            //word embedding
            int32_t boundary = uniform(model.rng);
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    model.update(line[w], line[w + c], lr);
                }
            }
            //entity embedding
            for (uint32_t l = 0; l < label.size(); l++) {
                model.update(label[l] + data_->nwords(), line[w], lr);
            }
        }
    }


    void controller::saveModel(std::string name) {
        std::string path =args_->output + name  + ".bin";
        std::ofstream ofs(path, std::ofstream::binary);
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