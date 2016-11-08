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
#include <memory>

namespace entity2vec {
    struct entry_word {
        std::string word;
        uint32_t count;
        int64_t prod_id;
        //std::vector<uint32_t> subwords;
    };

    struct entry_prod{
        std::string prod;
        uint32_t count;
        uint32_t word_count;

        std::vector<int64_t> word2idx_;
        std::vector<std::string> idx2words_;
    };

    class data {
    private:

        std::string UNK = "<UNK>";

        int64_t cur_prod_id;
        std::shared_ptr<args> args_;
    public:
        static const int64_t VOCAB_HASH_SIZE = 30000000;
        static const int64_t SUB_VOCAB_HASH_SIZE = 10000;
        static const int64_t PROD_HASH_SIZE = 500000;
        static const int32_t MAX_LINE_SIZE = 1024;

        uint64_t word_size_;
        uint64_t prod_size_;

        std::vector<int64_t> word2idx_;
        std::vector<entry_word> idx2words_;
        std::vector<int64_t> prod2idx_;
        std::vector<entry_prod> idx2prod_;

        explicit data(std::shared_ptr<args> args);

        uint64_t hash(const std::string& str) const;
        int64_t getWordHash(const std::string &word) const;
        int64_t getProdHash(const std::string &prod) const;
        int64_t getWordHashRegardingProd(const std::string &word, const std::string &prod) const;
        int64_t getWordId(const std::string& word) const;
        int64_t getProdId(const std::string& prod) const;
        bool checkWordInProd(const std::string &word, const std::string &prod) const;
        bool checkWordInProd(int64_t wid, int64_t pid) const;
        std::string getWord(uint32_t i) const;
        void addWord(const std::string& word);
        void addProd(const std::string& prod);
        std::vector<uint32_t> getWordCounts();

        void readFromFile(std::istream &in);
        uint32_t getLine(std::istream& in, std::vector<int64_t>& words, std::vector<int64_t>& labels, std::minstd_rand& rng) const;

        uint32_t nwords();
        uint32_t nprods();

        void threshold(uint32_t t);

        void save(std::ostream& out) const;
        void load(std::istream& in);
    };



}


#endif //TRAIN_DATA_H
