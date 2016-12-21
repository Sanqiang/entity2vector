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
#include "args.h"
#include <regex>
#include <unordered_set>

namespace entity2vec{

    void controller::train(std::shared_ptr<args> args) {
        args_ = args;

        if(args_->load_model_flag){
            loadModel(args->load_model);
        }else{
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

            if(args_->prod_flag && args_->tag_flag){
                input_ = std::make_shared<matrix>(data_->nwords() + data_->nprods() + data_->ntags(), args_->dim);
                input_->uniform(1.0 / args_->dim);

                output_ = std::make_shared<matrix>(data_->nwords() + data_->nprods() + data_->ntags(), args_->dim);
                output_->zero();
            }else if(args_->prod_flag){
                input_ = std::make_shared<matrix>(data_->nwords() + data_->nprods(), args_->dim);
                input_->uniform(1.0 / args_->dim);

                output_ = std::make_shared<matrix>(data_->nwords() + data_->nprods() + data_->ntags(), args_->dim);
                output_->zero();
            }
            else{
                input_ = std::make_shared<matrix>(data_->nwords(), args_->dim);
                input_->uniform(1.0 / args_->dim);

                output_ = std::make_shared<matrix>(data_->nwords(), args_->dim);
                output_->zero();
            }
            model_ = std::make_shared<model>(input_, output_, args_,data_, 0);
            if(args_->pretraining_flag) {
                std::cout << "start reading pretraining file" << std::endl;
                populate_pretraining();
                std::cout << "finish reading pretraining file" << std::endl;
            }
        }

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
        std::string path = args_->input_data_pattern;
        path.replace(path.find("{i}"), std::string("{i}").size(),std::to_string(threadId));
        //std::string path = args_->input_data;

        std::ifstream ifs(path);
        std::cout<<"start trainThread: "<< threadId <<  ":" <<path<<std::endl;

        model model(input_, output_, args_, data_, threadId);
        model.initWordNegSampling();

        std::vector<int64_t> line;
        std::vector<int64_t> prods;
        std::vector<int64_t> tags;
        const uint32_t ntokens = data_->nwords();
        uint32_t localTokenCount = 0;
        uint32_t loop = 0;
        while (tokenCount < args_->epoch * ntokens){
            real progress = real(tokenCount) / (args_->epoch * ntokens);
            real lr = args_->lr * (1.0 - progress);
            localTokenCount += data_->getLine(ifs, line, prods, tags, model.rng);
            skipgram(model, lr, line, prods, tags);

            tokenCount += localTokenCount;
            localTokenCount = 0;
            if (loop++ % 1000 == 0 && threadId == 0 && args_->verbose > 1) {
                printInfo(progress, model.getLoss());
                saveModel(args_->load_model + std::to_string(floor(progress*args_->epoch)));
                saveVectors(args_->load_model + std::to_string(floor(progress*args_->epoch)));
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
        int64_t cur_vector_idx = 0, cur_word_idx;
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
                        output_->setValue(cur_word_idx, cur_vector_idx++, stod(word));
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
        std::vector<std::pair<real, int>> pairs = input_->findSimilarRow(i, k, 0, data_->nwords()+data_->nprods()+data_->ntags()-1);

        std::cout << "" <<word<< " : ";
        for (auto it = pairs.begin(); it != pairs.end(); ++it){
            int64_t id = it->second;
            if(0 == model_->checkIndexType(id)){
                std::cout << data_->getWord(id) << "\t";
            }else if(1 == model_->checkIndexType(id)){
                std::cout << data_->getProd(id - data_->nwords()) << "\t";
            }else if(2 == model_->checkIndexType(id)){
                std::cout << data_->getTag(id - data_->nwords() - data_->nprods()) << "\t";
            }

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
        if(args_->verbose > 2) {
            printWords("steak", 10);
            printWords("seafood", 10);
            printWords("grill", 10);
            printWords("chines", 10);
            printWords("bbq", 10);
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::flush;
    }

    void controller::skipgram(model& model, real lr, const std::vector<int64_t> &line, const std::vector<int64_t> &prods, const std::vector<int64_t> &tags) {
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (int64_t w = 0; w < line.size(); w++) {
            if(line[w] < 0){
                continue;
            }
            //word embedding
            int32_t boundary = uniform(model.rng);
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    if(line[w + c] < 0){
                        continue;
                    }
                    //model.update(line[w], line[w + c], lr);
                }
            }
            //entity embedding
            if(args_->prod_flag) {
                //word - prod
                for (int64_t l = 0; l < prods.size(); l++) {
                    if(prods[l]  < 0){
                        continue;
                    }
                    model.update(model_->transform_dic2matrix(prods[l], 1), line[w], lr);
//                    model.update(line[w],model_->transform_dic2matrix(prods[l], 1), lr);
                }
            }

            if(args_->tag_flag){
                //word - tag
                for (int64_t l = 0; l < tags.size(); l++) {
                    if(tags[l]  < 0){
                        continue;
                    }
//                    model.update(line[w], model_->transform_dic2matrix(tags[l], 2), lr);
//                    model.update(model_->transform_dic2matrix(tags[l], 2), line[w], lr);
                }
            }
        }
        if(args_->tag_flag) {
            //tag - prod
            for (int64_t k = 0; k < prods.size(); k++) {
                if(prods[k] < 0){ continue;}
                for (int64_t l = 0; l < tags.size(); l++) {
                    if(tags[l] < 0 || prods[k] < 0){ continue;}
//                    model.update(model_->transform_dic2matrix(tags[l], 2), model_->transform_dic2matrix(prods[k], 1) , lr);
//                    model.update(model_->transform_dic2matrix(prods[k], 1) ,model_->transform_dic2matrix(tags[l], 2), lr);
                }
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
        //model_->save(ofs);
        ofs.close();
    }

    void controller::loadModel(std::istream &in) {
        data_ = std::make_shared<data>(args_);
        input_ = std::make_shared<matrix>();
        output_ = std::make_shared<matrix>();
        args_->load(in);
        data_->load(in);
        input_->load(in);
        output_->load(in);
        model_ = std::make_shared<model>(input_, output_, args_,data_, 0);
//        model_->initWordNegSampling();
//        model_->load(in);
    }

    void controller::loadModel(const std::string &name) {
        std::string path =args_->output + name  + ".bin";
        args_ = std::make_shared<args>();
        std::ifstream ifs(path, std::ifstream::binary);
        if (!ifs.is_open()) {
            std::cerr << "Model file cannot be opened for loading!" << std::endl;
            exit(EXIT_FAILURE);
        }
        loadModel(ifs);
        ifs.close();
    }

    void controller::saveVectors(const std::string &name) {
        std::string path = args_->output + name + ".word.vec";
        std::ofstream ofs_word(path, std::ofstream::binary);
        if (!ofs_word.is_open()) {
            std::cout << "Error opening file for saving vectors." << std::endl;
            exit(EXIT_FAILURE);
        }
        std::ofstream ofs_prod(args_->output + name + ".prod.vec");
        if (!ofs_prod.is_open()) {
            std::cout << "Error opening file for saving vectors." << std::endl;
            exit(EXIT_FAILURE);
        }
        std::ofstream ofs_tag(args_->output + name + ".tag.vec");
        if (!ofs_tag.is_open()) {
            std::cout << "Error opening file for saving vectors." << std::endl;
           exit(EXIT_FAILURE);
        }
        saveModel(ofs_word, ofs_prod, ofs_tag);
        ofs_word.close();
        ofs_prod.close();
        ofs_tag.close();
    }

    void controller::saveModel(std::ostream &ofs_word, std::ostream &ofs_prod, std::ostream &ofs_tag) {
        ofs_word << data_->nwords() << " " << args_->dim << std::endl;
        vector vec_word(args_->dim);
        for (uint32_t i = 0; i < data_->nwords(); i++) {
            std::string word = data_->getWord(i);
            for (uint32_t j = 0; j < args_->dim; ++j) {
                vec_word.setData(input_->data_[i*args_->dim + j],j);
            }
            vec_word.normalize();

            ofs_word << word.data() << " ";
            for (uint32_t j = 0; j < args_->dim; ++j) {
                ofs_word << vec_word.data_[j];
                if(j == args_->dim - 1){
                    ofs_word << "\n";
                }else{
                    ofs_word << " ";
                }
            }
        }

        ofs_prod << data_->nprods() << " " << args_->dim << std::endl;
        vector vec_prod(args_->dim);
        for (int32_t i = 0; i < data_->nprods(); i++) {
            std::string prod = data_->getProd(i);
            for (uint32_t j = 0; j < args_->dim; ++j) {
                vec_prod.setData(input_->data_[i*args_->dim + j],j);
            }
            vec_prod.normalize();
            ofs_prod << prod << " ";
            for (uint32_t j = 0; j < args_->dim; ++j) {
                ofs_prod << vec_prod.data_[j];
                if(j == args_->dim - 1){
                    ofs_prod << "\n";
                }else{
                    ofs_prod << " ";
                }
            }
        }

        ofs_tag << data_->ntags() << " " << args_->dim << std::endl;
        vector vec_tag(args_->dim);
        for (int32_t i = 0; i < data_->ntags(); i++) {
            std::string tag = data_->getTag(i);
            for (uint32_t j = 0; j < args_->dim; ++j) {
                vec_tag.setData(input_->data_[i*args_->dim + j],j);
            }
            vec_tag.normalize();

            ofs_tag << tag << " ";
            for (uint32_t j = 0; j < args_->dim; ++j) {
                ofs_tag << vec_tag.data_[j];
                if(j == args_->dim - 1){
                    ofs_tag << "\n";
                }else{
                    ofs_tag << " ";
                }
            }
        }
    }


}