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
        epoch = 1000000000000;
        minCount = 5;
        neg = 5;
        wordNgrams = 1;
        minn = 0;
        maxn = 6;
        thread = 1;
        lrUpdateRate = 0;
        t = 1e-4;
        label = "__label__";
        verbose = 2;
        pretrainedVectors = "";
        std::string base =  "/home/sanqiang/";//getenv("HOME");
        input_data = base + "/data/yelp/review_processed.txt";
        input_pretrain =  base + "/data/glove/glove.processed.840B.300d.txt";
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
    }

}