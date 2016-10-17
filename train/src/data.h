//
// Created by Sanqiang Zhao on 10/9/16.
//

#ifndef TRAIN_DATA_H
#define TRAIN_DATA_H

#include "args.h"
#include <unordered_map>
#include <vector>
#include <string>

namespace entity2vec {
    struct entry {
        std::string word;
        uint64_t count;
        uint32_t prod_id;
    };

    class data {
    private:
        static const uint32_t VOCAB_HASH_SIZE = 30000000;
        static const uint32_t PROD_HASH_SIZE = 50000;
        static const uint32_t MAX_LINE_SIZE = 1024;

        uint8_t cur_mode = 0; //0:prod 1:text // 0:user 1:prod 2:rating 3:text
        uint64_t cur_prod_id;
        uint64_t word_size_;
        uint64_t prod_size_;

        std::vector<uint32_t> word2idx_;
        std::vector<entry> idx2words_;
        std::vector<uint32_t> prod2idx_;
        std::vector<std::string> idx2prod_;

    public:
        explicit data();
        explicit data(std::shared_ptr<args> args);

        uint32_t hash(const std::string& str) const;
        uint32_t findWord(const std::string &word) const;
        uint32_t findProd(const std::string &prod) const;
        void addWord(const std::string& word);
        void addProd(const std::string& prod);
        std::vector<uint64_t> getCounts();
        void readFromFile(std::istream &in);
        uint64_t nwords();
    };



}


#endif //TRAIN_DATA_H
