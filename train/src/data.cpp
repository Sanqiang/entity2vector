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

namespace entity2vec {

    data::data(std::shared_ptr<args> args) {
        args_ = args;
        cur_mode = 0;
        word_size_ = 0;
        prod_size_ = 0;
        word2idx_.resize(VOCAB_HASH_SIZE);
        for (uint32_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = 0;
        }
        prod2idx_.reserve(PROD_HASH_SIZE);
        for (uint32_t i = 0; i < PROD_HASH_SIZE; i++) {
            prod2idx_[i] = 0;
        }

        //for UNK
        addWord("<UNK>");
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
        if (word2idx_[h] == 0) {
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
        if (prod2idx_[h] == 0) {
            prod2idx_[h] = prod_size_++;
            idx2prod_.push_back(prod);
        }else{
        }
    }

    uint32_t data::findProd(const std::string &prod) const {
        uint32_t h = hash(prod) % PROD_HASH_SIZE;
        while (prod2idx_[h] != 0 && idx2prod_[prod2idx_[h]] != prod) {
            h = (h + 1) % PROD_HASH_SIZE;
        }
        return h;
    }

    uint32_t data::findWord(const std::string &word) const {
        uint32_t h = hash(word) % VOCAB_HASH_SIZE;
        while (word2idx_[h] != 0 && idx2words_[word2idx_[h]].word != word) {
            h = (h + 1) % VOCAB_HASH_SIZE;
        }
        return h;
    }

    uint32_t data::getWordId(const std::string &word) const {
        uint32_t h = findWord(word);
        return word2idx_[h];
    }

    uint32_t data::getProdId(const std::string &prod) const {
        uint32_t h = findProd(prod);
        return prod2idx_[h];
    }

    std::string data::getWord(uint32_t i) const {
        return idx2words_[i].word;
    }

    void data::readFromFile(std::istream &in) {
        std::string word;
        char c;
        std::streambuf& sb = *in.rdbuf();
        while ((c = sb.sbumpc()) != EOF) {
            if (c == ' ' || c == '\t' || c == '\v' || c == '\n') {
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

    uint32_t data::getLine(std::istream &in, std::vector<uint32_t> &words, std::vector<uint32_t> &labels,
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
                        uint32_t wid = getWordId(word);
                        words.push_back(wid);
                        //if (wid < 0) continue;
                        ntokens++;
                    }else if(cur_mode == 0){
                        uint32_t pid = getProdId(word);
                        labels.push_back(pid);
                    }
                }

                if(c == '\t'){
                    cur_mode++;
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

    std::vector<uint32_t> data::getCounts() {
        std::vector<uint32_t> counts;
        for (auto& w : idx2words_) {
            counts.push_back(w.count);
        }
        return counts;
    }

    void data::threshold(uint64_t t) {
        sort(idx2words_.begin(), idx2words_.end(), [](const entry& e1, const entry& e2) {
            return e1.count > e2.count;
        });
        idx2words_.erase(remove_if(idx2words_.begin(), idx2words_.end(), [&](const entry& e) {
            return e.count < t;
        }), idx2words_.end());
        idx2words_.shrink_to_fit();
        word_size_ = 0;
        for (int32_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = 0;
        }
        for (auto it = idx2words_.begin(); it != idx2words_.end(); ++it) {
            uint32_t h = findWord(it->word);
            word2idx_[h] = word_size_++;
        }
    }

    void data::save(std::ostream &out) const {
        out.write((char*) &word_size_, sizeof(uint32_t));
        out.write((char*) &prod_size_, sizeof(uint32_t));
        for (uint32_t i = 0; i < word_size_; i++) {
            entry e = idx2words_[i];
            out.write(e.word.data(), e.word.size() * sizeof(char));
            out.put(0);
            out.write((char*) &(e.count), sizeof(uint32_t));
            out.write((char*) &(e.prod_id), sizeof(uint32_t));
        }
        for (uint32_t i = 0; i < prod_size_; i++) {
            out.write(idx2prod_[i].data(), idx2prod_[i].size() * sizeof(char));
            out.put(0);
        }
    }

    void data::load(std::istream &in) {
        idx2words_.clear();
        idx2prod_.clear();
        for (uint32_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = 0;
        }
        prod2idx_.reserve(PROD_HASH_SIZE);
        for (uint32_t i = 0; i < PROD_HASH_SIZE; i++) {
            prod2idx_[i] = 0;
        }
        in.read((char*) &word_size_, sizeof(uint32_t));
        in.read((char*) &prod_size_, sizeof(uint32_t));

        for (uint32_t i = 0; i < word_size_; i++) {
            char c;
            entry e;
            while ((c = in.get()) != 0) {
                e.word.push_back(c);
            }
            in.read((char*) &e.count, sizeof(uint32_t));
            in.read((char*) &e.prod_id, sizeof(uint32_t));
            idx2words_.push_back(e);
            word2idx_[findWord(e.word)] = i;
        }
        for (uint32_t i = 0; i < prod_size_; i++) {
            char c;
            std::string prod;
            while ((c = in.get()) != 0) {
                prod.push_back(c);
            }
            idx2prod_.push_back(prod);
            prod2idx_[findWord(prod)] = i;
        }

    }
}
