//
// Created by Sanqiang Zhao on 10/9/16.
//

#ifndef TRAIN_DATA_H
#define TRAIN_DATA_H

#include "args.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <random>

namespace entity2vec {
    struct entry {
        std::string word;
        uint32_t count;
        uint32_t prod_id;
        std::vector<uint32_t> subwords;
    };

    class data {
    private:
        static const uint32_t VOCAB_HASH_SIZE = 30000000;
        static const uint32_t PROD_HASH_SIZE = 50000;
        static const uint32_t MAX_LINE_SIZE = 1024;

        uint8_t cur_mode = 0; //0:prod 1:text // 0:user 1:prod 2:rating 3:text
        uint32_t cur_prod_id;
        uint32_t word_size_;
        uint32_t prod_size_;

        std::vector<uint32_t> word2idx_;
        std::vector<entry> idx2words_;
        std::vector<uint32_t> prod2idx_;
        std::vector<std::string> idx2prod_;

        std::shared_ptr<args> args_;

    public:
        explicit data(std::shared_ptr<args> args);

        uint32_t hash(const std::string& str) const;
        uint32_t findWord(const std::string &word) const;
        uint32_t findProd(const std::string &prod) const;
        uint32_t getWordId(const std::string& word) const;
        uint32_t getProdId(const std::string& prod) const;
        std::string getWord(uint32_t i) const;
        void addWord(const std::string& word);
        void addProd(const std::string& prod);
        std::vector<uint32_t> getCounts();
        const std::vector<uint32_t>& getNgrams(uint32_t i) const;

        void readFromFile(std::istream &in);
        uint32_t getLine(std::istream& in, std::vector<uint32_t>& words, std::vector<uint32_t>& labels, std::minstd_rand& rng) const;

        uint32_t nwords();
        uint32_t nprods();

        void save(std::ostream& out) const;
        void load(std::istream& in);
    };



}


#endif //TRAIN_DATA_H
