//
// Created by Sanqiang Zhao on 10/9/16.
//

#include "data.h"

#include <iostream>

namespace entity2vec {

    data::data() {
        cur_mode = 0;
        word_size_ = 0;
        prod_size_ = 0;
        word2idx_.resize(VOCAB_HASH_SIZE);
        for (uint32_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = -1;
        }
        prod2idx_.reserve(PROD_HASH_SIZE);
        for (uint32_t i = 0; i < PROD_HASH_SIZE; i++) {
            prod2idx_[i] = -1;
        }
    }

    data::data(std::shared_ptr<args> args) {
        cur_mode = 0;
        word_size_ = 0;
        prod_size_ = 0;
        word2idx_.resize(VOCAB_HASH_SIZE);
        for (uint32_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = -1;
        }
        prod2idx_.reserve(PROD_HASH_SIZE);
        for (uint32_t i = 0; i < PROD_HASH_SIZE; i++) {
            prod2idx_[i] = -1;
        }
    }

    uint32_t data::hash(const std::string &str) const {
        uint32_t h = 2166136261;
        for (size_t i = 0; i < str.size(); i++) {
            h = h ^ uint32_t(str[i]);
            h = h * 16777619;
        }
        return h;
    }

    void data::addWord(const std::string &word) {
        uint32_t h = findWord(word);
        if (word2idx_[h] == -1) {
            entry e;
            e.word = word;
            e.prod_id = cur_prod_id;
            e.count = 1;
            word2idx_[h] = word_size_++;
            idx2words_.push_back(e);
        } else {
            idx2words_[word2idx_[h]].count++;
        }
    }

    void data::addProd(const std::string &prod) {
        uint32_t h = findProd(prod);
        if (prod2idx_[h] == -1) {
            prod2idx_[h] = prod_size_++;
            idx2prod_.push_back(prod);
        }else{
        }
    }

    uint32_t data::findProd(const std::string &prod) const {
        uint32_t h = hash(prod) % PROD_HASH_SIZE;
        while (prod2idx_[h] != -1 && idx2prod_[prod2idx_[h]] != prod) {
            h = (h + 1) % PROD_HASH_SIZE;
        }
        return h;
    }

    uint32_t data::findWord(const std::string &word) const {
        uint32_t h = hash(word) % VOCAB_HASH_SIZE;
        while (word2idx_[h] != -1 && idx2words_[word2idx_[h]].word != word) {
            h = (h + 1) % VOCAB_HASH_SIZE;
        }
        return h;
    }

    void data::readFromFile(std::istream &in) {
        std::string word;
        char c;
        std::streambuf& sb = *in.rdbuf();
        while ((c = sb.sbumpc()) != EOF) {
            if (c == ' ' || c == '\t' || c == '\v') {
                if(!word.empty()){
                    if(cur_mode == 0){
                        addProd(word);
                        cur_prod_id = prod2idx_[findProd(word)];
                    }else if(cur_mode == 1){
                        addWord(word);
                    }
                }

                if(c == '\t'){
                    cur_mode++;
                }else if(c == '\v'){

                }
                word.clear();
            }else{
                word.push_back(c);
            }
        }
    }

    uint64_t data::nwords() {
        return word_size_;
    }

    std::vector<uint64_t> data::getCounts() {
        std::vector<uint64_t> counts;
        for (auto& w : idx2words_) {
            counts.push_back(w.count);
        }
        return counts;
    }
}
