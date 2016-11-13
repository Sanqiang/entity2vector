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
    };

    struct entry_prod{
        std::string prod;
        uint32_t count;
        uint32_t word_count;
    };

    struct entry_tag{
        std::string tag;
        uint32_t count;
        uint32_t word_count;
    };

    class data {
    private:

        std::string UNK = "<UNK>";

        std::string cur_prod;
        std::vector<std::string> cur_tags;

        std::shared_ptr<args> args_;
    public:
        static const int64_t VOCAB_HASH_SIZE = 100000000;
        static const int64_t TAG_HASH_SIZE = 10000;
        static const int64_t PROD_HASH_SIZE = 10000;

        static const int64_t WORD_PROD_HASH_SIZE = 100000000;
        static const int64_t WORD_TAG_HASH_SIZE = 100000000;
        static const int64_t TAG_PROD_HASH_SIZE = 100000000;


        //static const int32_t MAX_LINE_SIZE = 1024;

        uint64_t word_size_;
        uint64_t prod_size_;
        uint64_t tag_size_;

        std::vector<int64_t> word2idx_;
        std::vector<entry_word> idx2words_;
        std::vector<int64_t> prod2idx_;
        std::vector<entry_prod> idx2prod_;
        std::vector<int64_t> tag2idx_;
        std::vector<entry_tag> idx2tag_;

        std::vector<std::string> idx2cor_word_prod_;
        std::vector<std::string> idx2cor_word_tag_;
        std::vector<std::string> idx2cor_tag_prod_;
        std::vector<int64_t> cor_word_prod2idx_;
        std::vector<int64_t> cor_word_tag2idx_;
        std::vector<int64_t> cor_tag_prod2idx_;

        explicit data(std::shared_ptr<args> args);

        uint64_t hash(const std::string& str) const;
        int64_t getWordHash(const std::string &word) const;
        int64_t getProdHash(const std::string &prod) const;
        int64_t getTagHash(const std::string &tag) const;
        int64_t getWordId(const std::string& word) const;
        int64_t getProdId(const std::string& prod) const;
        int64_t getTagId(const std::string& tag) const;

        int64_t getHashCorPair(const std::string &key, uint8_t mode) const; //mode 1: word-prod 2: word-tag 3: tag-prod
        int64_t getHashCorPair(const std::string &pair1, const std::string &pair2, uint8_t mode) const; //mode 1: word-prod 2: word-tag 3: tag-prod
        bool checkCorPair(const std::string &pair1, const std::string &pair2, uint8_t mode) const;
        void addCorPair(const std::string &pair1, const std::string &pair2, uint8_t mode);

        void addWord(const std::string& word);
        void addProd(const std::string& prod);
        void addTag(const std::string& tag);

        std::vector<uint32_t> getWordCounts();

        void readFromFile(std::istream &in);
        uint32_t getLine(std::istream& in, std::vector<int64_t>& words, std::vector<int64_t>& prods, std::vector<int64_t>& tags, std::minstd_rand& rng) const;

        std::string getWord(uint32_t i) const;
        uint32_t nwords();
        uint32_t nprods();

        void threshold(uint32_t t);

        void save(std::ostream& out) const;
        void load(std::istream& in);
    };
}


#endif //TRAIN_DATA_H
