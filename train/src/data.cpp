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
        addProd("<UNK>");
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
        uint32_t h = getWordHash(word);

        if (word2idx_[h] == 0) {
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
        entry_prod e_prod = idx2prod_[cur_prod_id];
        h = getWordHashRegardingProd(word, e_prod.prod);
        if(e_prod.word2idx_[h] != 0) {
            e_prod.idx2words_.push_back(word);
            e_prod.word2idx_[h] = e_prod.idx2words_.size()-1;
            e_prod.word_count++;
        }


    }

    void data::addProd(const std::string &prod) {
        uint32_t h = getProdHash(prod);
        if (prod2idx_[h] == 0) {
            entry_prod e;
            e.prod = prod;
            e.count = 1;
            e.word2idx_.resize(SUB_VOCAB_HASH_SIZE);
            for (uint32_t i = 0; i < SUB_VOCAB_HASH_SIZE; i++) {
                e.word2idx_[i] = 0;
            }

            prod2idx_[h] = prod_size_++;
            idx2prod_.push_back(e);
        }else{
            idx2prod_[prod2idx_[h]].count++;
        }
    }

    uint32_t data::getProdHash(const std::string &prod) const {
        uint32_t h = hash(prod) % PROD_HASH_SIZE;
        while (prod2idx_[h] != 0 && idx2prod_[prod2idx_[h]].prod != prod) {
            h = (h + 1) % PROD_HASH_SIZE;
        }
        return h;
    }

    uint32_t data::getWordHash(const std::string &word) const {
        uint32_t h = hash(word) % VOCAB_HASH_SIZE;
        while (word2idx_[h] != 0 && idx2words_[word2idx_[h]].word != word) {
            h = (h + 1) % VOCAB_HASH_SIZE;
        }
        return h;
    }

    uint32_t data::getWordHashRegardingProd(const std::string &word, const std::string &prod) const {
        uint32_t h = hash(word) % SUB_VOCAB_HASH_SIZE;
        entry_prod e = idx2prod_[getProdId(prod)];
        while (word2idx_[h] != 0 && idx2words_[e.word2idx_[h]].word != word) {
            h = (h + 1) % SUB_VOCAB_HASH_SIZE;
        }
        return h;
    }

    bool data::checkWordInProd(const std::string &word, const std::string &prod) const {
        uint32_t h = getWordHashRegardingProd(word, prod);
        uint32_t pid = getProdId(prod);
        return idx2prod_[pid].word2idx_[h] != 0;
    }

    bool data::checkWordInProd(uint32_t wid, uint32_t pid) const {
        return checkWordInProd(idx2words_[wid].word, idx2prod_[pid].prod);
    }

    uint32_t data::getWordId(const std::string &word) const {
        uint32_t h = getWordHash(word);
        return word2idx_[h];
    }

    uint32_t data::getProdId(const std::string &prod) const {
        uint32_t h = getProdHash(prod);
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
                        cur_prod_id = prod2idx_[getProdHash(word)];
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
        //vanish <UNK>
        idx2words_[0].count = 0;
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

    std::vector<uint32_t> data::getWordCounts() {
        std::vector<uint32_t> counts;
        for (auto& w : idx2words_) {
            counts.push_back(w.count);
        }
        return counts;
    }

    void data::threshold(uint64_t t) {
        sort(idx2words_.begin(), idx2words_.end(), [](const entry_word& e1, const entry_word& e2) {
            return e1.count > e2.count;
        });
        idx2words_.erase(remove_if(idx2words_.begin(), idx2words_.end(), [&](const entry_word& e) {
            return e.count < t;
        }), idx2words_.end());
        idx2words_.shrink_to_fit();
        word_size_ = 0;
        for (int32_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = 0;
        }
        for (auto it = idx2words_.begin(); it != idx2words_.end(); ++it) {
            uint32_t h = getWordHash(it->word);
            word2idx_[h] = word_size_++;
        }
    }

    void data::save(std::ostream &out) const {
        out.write((char*) &word_size_, sizeof(uint32_t));
        out.write((char*) &prod_size_, sizeof(uint32_t));
        for (uint32_t i = 0; i < word_size_; i++) {
            entry_word e = idx2words_[i];
            out.write(e.word.data(), e.word.size() * sizeof(char));
            out.put(0);
            out.write((char*) &(e.count), sizeof(uint32_t));
            out.write((char*) &(e.prod_id), sizeof(uint32_t));
        }
//todo serialize prod
//        for (uint32_t i = 0; i < prod_size_; i++) {
//            out.write(idx2prod_[i].data(), idx2prod_[i].size() * sizeof(char));
//            out.put(0);
//        }
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
            entry_word e;
            while ((c = in.get()) != 0) {
                e.word.push_back(c);
            }
            in.read((char*) &e.count, sizeof(uint32_t));
            in.read((char*) &e.prod_id, sizeof(uint32_t));
            idx2words_.push_back(e);
            word2idx_[getWordHash(e.word)] = i;
        }
        for (uint32_t i = 0; i < prod_size_; i++) {
            char c;
            std::string prod;
            while ((c = in.get()) != 0) {
                prod.push_back(c);
            }
            //todo serialize prod
            //idx2prod_.push_back(prod);
            prod2idx_[getWordHash(prod)] = i;
        }

    }
}
