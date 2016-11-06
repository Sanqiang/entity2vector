//
// Created by Sanqiang Zhao on 10/9/16.
//

#include "data.h"
#include <memory>
#include <iostream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <cctype>
#include <stdio.h>
#include <string.h>

namespace entity2vec {

    data::data(std::shared_ptr<args> args) {
        args_ = args;
        word_size_ = 0;
        prod_size_ = 0;
        word2idx_.resize(VOCAB_HASH_SIZE);
        for (int64_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = -1;
        }
        prod2idx_.reserve(PROD_HASH_SIZE);
        for (int64_t i = 0; i < PROD_HASH_SIZE; i++) {
            prod2idx_[i] = -1;
        }
    }

    uint64_t data::hash(const std::string &str) const {
        uint64_t h = 2166136261;
        for (size_t i = 0; i < str.size(); i++) {
            h = h ^ uint32_t(str[i]);
            h = h * 16777619;
        }
        return h;
    }

    void data::addWord(const std::string &word) {
        int64_t h = getWordHash(word);

        if (word2idx_[h] == -1) {
            entry_word e;
            e.word = word;
            e.prod_id = cur_prod_id;
            e.count = 1;

            word2idx_[h] = word_size_++;
            idx2words_.push_back(e);
        } else {
            idx2words_[word2idx_[h]].count++;
        }

        //process entry_prod
        h = getWordHashRegardingProd(word, idx2prod_[cur_prod_id].prod);
        if(idx2prod_[cur_prod_id].word2idx_[h] == -1) {
            idx2prod_[cur_prod_id].idx2words_.push_back(word);
            idx2prod_[cur_prod_id].word2idx_[h] = idx2prod_[cur_prod_id].idx2words_.size()-1;
            idx2prod_[cur_prod_id].word_count++;
        }
    }

    void data::addProd(const std::string &prod) {
        int64_t h = getProdHash(prod);
        if (prod2idx_[h] == -1) {
            entry_prod e;
            e.prod = prod;
            e.count = 1;
            e.word_count = 0;
            e.word2idx_.resize(SUB_VOCAB_HASH_SIZE);
            for (int64_t i = 0; i < SUB_VOCAB_HASH_SIZE; i++) {
                e.word2idx_[i] = -1;
            }

            prod2idx_[h] = prod_size_++;
            idx2prod_.push_back(e);
        }else{
            idx2prod_[prod2idx_[h]].count++;
        }
    }

    int64_t data::getProdHash(const std::string &prod) const {
        int64_t h = hash(prod) % PROD_HASH_SIZE;
        while (prod2idx_[h] != -1 && idx2prod_[prod2idx_[h]].prod != prod) {
            h = (h + 1) % PROD_HASH_SIZE;
        }
        return h;
    }

    int64_t data::getWordHash(const std::string &word) const {
        int64_t h = hash(word) % VOCAB_HASH_SIZE;
        while (word2idx_[h] != -1 && idx2words_[word2idx_[h]].word != word) {
            h = (h + 1) % VOCAB_HASH_SIZE;
        }
        return h;
    }

    int64_t data::getWordHashRegardingProd(const std::string &word, const std::string &prod) const {
        int64_t h = hash(word) % SUB_VOCAB_HASH_SIZE;
        entry_prod e = idx2prod_[getProdId(prod)];
        while (e.word2idx_[h] != -1 && e.idx2words_[e.word2idx_[h]] != word) {
            h = (h + 1) % SUB_VOCAB_HASH_SIZE;
        }
        return h;
    }

    bool data::checkWordInProd(const std::string &word, const std::string &prod) const {
        int64_t h = getWordHashRegardingProd(word, prod);
        int64_t pid = getProdId(prod);
        return idx2prod_[pid].word2idx_[h] != -1;
    }

    bool data::checkWordInProd(int64_t wid, int64_t pid) const {
        return checkWordInProd(idx2words_[wid].word, idx2prod_[pid].prod);
    }

    int64_t data::getWordId(const std::string &word) const {
        int64_t h = getWordHash(word);
        return word2idx_[h];
    }

    int64_t data::getProdId(const std::string &prod) const {
        int64_t h = getProdHash(prod);
        return prod2idx_[h];
    }

    std::string data::getWord(uint32_t i) const {
        return idx2words_[i].word;
    }

    void data::readFromFile(std::istream &in) {
        std::string word;
        char c;
        uint8_t cur_mode = 0; //0:prod 1:text // 0:user 1:prod 2:rating 3:text
        std::streambuf& sb = *in.rdbuf();
        while ((c = sb.sbumpc()) != EOF) {
            if (c == ' ' || c == '\t' || c == '\v' || c == '\n') {
                if(!word.empty()){
                    if(cur_mode == 0){
                        addProd(word);
                        cur_prod_id = prod2idx_[getProdHash(word)];
                    }else if(cur_mode == 1){
                        addWord(word);
                    }
                }

                if(c == '\t'){
                    cur_mode = 1;
                }else if(c == '\v'){

                }else if(c == '\n'){
                    cur_mode = 0;
                }
                word.clear();
            }else{
                word.push_back(c);
            }
        }
        threshold(args_->minCount);
        if (args_->verbose > 0) {
            std::cout << "Number of words:  " << word_size_ << std::endl;
            std::cout << "Number of prods:  " << prod_size_ << std::endl;
        }
    }

    uint32_t data::getLine(std::istream &in, std::vector<int64_t> &words, std::vector<int64_t> &labels,
                           std::minstd_rand &rng) const {
        std::uniform_real_distribution<> uniform(0, 1);
        std::string token;
        uint32_t ntokens = 0;
        words.clear();
        labels.clear();
        uint8_t cur_mode = 0;
        std::string word;
        char c;
        std::streambuf& sb = *in.rdbuf();
        while ((c = sb.sbumpc()) != EOF) {
            if (c == ' ' || c == '\t' || c == '\v' || c == '\n') {
                if(!word.empty()){
                    if(cur_mode == 1){
                        int64_t wid = getWordId(word);
                        words.push_back(wid);
                        //if (wid < 0) continue;
                        ntokens++;
                    }else if(cur_mode == 0){
                        int64_t pid = getProdId(word);
                        labels.push_back(pid);
                    }
                }

                if(c == '\t'){
                    cur_mode = 1;
                }else if(c == '\v'){

                }else if(c == '\n'){
                    cur_mode = 0;
                    break;
                }
                word.clear();
            }else{
                word.push_back(c);
            }
        }
        if((c = sb.sbumpc()) == EOF){
            in.clear();
            in.seekg(std::streampos(0));
        }
        return ntokens;
    }

    uint32_t data::nwords() {
        return word_size_;
    }

    uint32_t data::nprods() {
        return prod_size_;
    }

    std::vector<uint32_t> data::getWordCounts() {
        std::vector<uint32_t> counts;
        for (auto& w : idx2words_) {
            counts.push_back(w.count);
        }
        return counts;
    }

    void data::threshold(uint32_t t) {
        sort(idx2words_.begin(), idx2words_.end(), [](const entry_word& e1, const entry_word& e2) {
            return e1.count > e2.count;
        });
        idx2words_.erase(remove_if(idx2words_.begin(), idx2words_.end(), [&](const entry_word& e) {
            return e.count < t;
        }), idx2words_.end());
        idx2words_.shrink_to_fit();
        word_size_ = 0;
        for (int64_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = -1;
        }
        for (auto it = idx2words_.begin(); it != idx2words_.end(); ++it) {
            uint32_t h = getWordHash(it->word);
            word2idx_[h] = word_size_++;
        }
    }

    void data::save(std::ostream &out) const {
        out.write((char*) &word_size_, sizeof(uint64_t));
        out.write((char*) &prod_size_, sizeof(uint64_t));
        for (uint64_t i = 0; i < word_size_; i++) {
            entry_word e = idx2words_[i];
            out.write(e.word.data(), e.word.size() * sizeof(char));
            out.put(0);
            out.write((char*) &(e.count), sizeof(uint32_t));
            out.write((char*) &(e.prod_id), sizeof(int64_t));
        }
        for (uint64_t i = 0; i < prod_size_; i++) {
            entry_prod e = idx2prod_[i];
            out.write(e.prod.data(), e.prod.size() * sizeof(char));
            out.put(0);
            out.write((char*) &(e.count), sizeof(uint32_t));
            out.write((char*) &(e.word_count), sizeof(uint32_t));

            for (uint32_t j = 0; j < e.word_count; j++) {
                out.write(e.idx2words_[j].data(), e.idx2words_[j].size() * sizeof(char));
                out.put(0);
            }
        }
    }

    void data::load(std::istream &in) {
        idx2words_.clear();
        idx2prod_.clear();
        word2idx_.resize(VOCAB_HASH_SIZE);
        for (int64_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = -1;
        }
        prod2idx_.reserve(PROD_HASH_SIZE);
        prod2idx_.reserve(PROD_HASH_SIZE);
        for (int64_t i = 0; i < PROD_HASH_SIZE; i++) {
            prod2idx_[i] = -1;
        }
        in.read((char*) &word_size_, sizeof(uint64_t));
        in.read((char*) &prod_size_, sizeof(uint64_t));

        for (uint32_t i = 0; i < word_size_; i++) {
            char c;
            entry_word e;
            while ((c = in.get()) != 0) {
                e.word.push_back(c);
            }
            in.read((char*) &e.count, sizeof(uint32_t));
            in.read((char*) &e.prod_id, sizeof(int64_t));
            idx2words_.push_back(e);
            word2idx_[getWordHash(e.word)] = i;
        }

        for (uint32_t i = 0; i < prod_size_; i++) {
            char c;
            entry_prod e;
            while ((c = in.get()) != 0) {
                e.prod.push_back(c);
            }

            in.read((char*) &e.count, sizeof(uint32_t));
            in.read((char*) &e.word_count, sizeof(uint32_t));

            e.word2idx_.resize(SUB_VOCAB_HASH_SIZE);
            for (int64_t k = 0; k < SUB_VOCAB_HASH_SIZE; k++) {
                e.word2idx_[k] = -1;
            }

            idx2prod_.push_back(e);
            prod2idx_[getProdHash(e.prod)] = i;

            for (uint32_t j = 0; j < e.word_count; j++) {

                std::string word;
                while ((c = in.get()) != 0) {
                    word.push_back(c);
                }

                e.idx2words_.push_back(word);
                e.word2idx_[getWordHashRegardingProd(word, e.prod)] = j;
            }


        }

    }
}
