//
// Created by Sanqiang Zhao on 10/15/16.
//

#include "args.h"

#include <fstream>
#include <string>

namespace entity2vec {
    args::args() {
        lr = 0.05;
        dim = 300;
        ws = 5;
        epoch = 1000;
        minCount = 0;
        neg = 5;
        minn = 0;
        maxn = 6;
        thread = 8;
        neg_trial = 100;
        lrUpdateRate = 0;
        t = 1e-4;
        label = "__label__";
        verbose = 3;
        pretrainedVectors = "";
        std::string base =  getenv("HOME");
        //input_data = base + "/data/aan/paper_processed_nostem_3.txt";
        input_data = base + "/data/yelp/review_processed_rest.txt";
        //input_data = base + "/data/yelp/review_processed_nostem_7.txt";
        input_data_pattern = base + "/data/yelp/review_processed_rest_{i}.txt";
        input_pretrain =  base + "/data/glove/glove.processed.840B.300d.txt";
        output = base + "/data/model/";

        mode_flag = 2;
        if(mode_flag == 0){
            prod_flag = 0;
            tag_flag = 0;
        }else if(mode_flag == 1){
            prod_flag = 1;
            tag_flag = 0;
        }else if(mode_flag == 2){
            prod_flag = 1;
            tag_flag = 1;
        }

        pretraining_flag = 1;

        load_model_flag = 0;
        load_model = "mode_flag_2_";
    }

    void args::save(std::ostream &out) {
        out.write((char*) &(dim), sizeof(uint32_t));
        out.write((char*) &(ws), sizeof(uint32_t));
        out.write((char*) &(epoch), sizeof(uint32_t));
        out.write((char*) &(minCount), sizeof(uint32_t));
        out.write((char*) &(neg), sizeof(uint32_t));
        out.write((char*) &(wordNgrams), sizeof(uint32_t));
//        out.write((char*) &(loss), sizeof(loss_name));
//        out.write((char*) &(model), sizeof(model_name));
//        out.write((char*) &(bucket), sizeof(int));
        out.write((char*) &(minn), sizeof(uint32_t));
        out.write((char*) &(maxn), sizeof(uint32_t));
        out.write((char*) &(lrUpdateRate), sizeof(uint32_t));
        out.write((char*) &(t), sizeof(double));
        out.write((char*) &(lr), sizeof(double));
    }

    void args::load(std::istream &in) {
        in.read((char*) &(dim), sizeof(uint32_t));
        in.read((char*) &(ws), sizeof(uint32_t));
        in.read((char*) &(epoch), sizeof(uint32_t));
        in.read((char*) &(minCount), sizeof(uint32_t));
        in.read((char*) &(neg), sizeof(uint32_t));
        in.read((char*) &(wordNgrams), sizeof(uint32_t));
//        in.read((char*) &(loss), sizeof(loss_name));
//        in.read((char*) &(model), sizeof(model_name));
//        in.read((char*) &(bucket), sizeof(int));
        in.read((char*) &(minn), sizeof(uint32_t));
        in.read((char*) &(maxn), sizeof(uint32_t));
        in.read((char*) &(lrUpdateRate), sizeof(uint32_t));
        in.read((char*) &(t), sizeof(double));
        in.read((char*) &(lr), sizeof(double));
    }

}