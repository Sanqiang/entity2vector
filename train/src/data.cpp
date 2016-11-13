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
        tag_size_ = 0;
        idx2words_.clear();
        idx2prod_.clear();
        idx2tag_.clear();
        idx2cor_word_prod_.clear();
        idx2cor_word_tag_.clear();
        idx2cor_tag_prod_.clear();

        word2idx_.resize(VOCAB_HASH_SIZE);
        for (int64_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = -1;
        }
        prod2idx_.reserve(PROD_HASH_SIZE);
        for (int64_t i = 0; i < PROD_HASH_SIZE; i++) {
            prod2idx_[i] = -1;
        }
        tag2idx_.reserve(TAG_HASH_SIZE);
        for (int64_t i = 0; i < TAG_HASH_SIZE; i++) {
            tag2idx_[i] = -1;
        }

        cor_word_prod2idx_.reserve(WORD_PROD_HASH_SIZE);
        for (int64_t i = 0; i < WORD_PROD_HASH_SIZE; i++) {
            cor_word_prod2idx_[i] = -1;
        }
        cor_word_tag2idx_.reserve(WORD_TAG_HASH_SIZE);
        for (int64_t i = 0; i < WORD_TAG_HASH_SIZE; i++) {
            cor_word_tag2idx_[i] = -1;
        }
        cor_tag_prod2idx_.reserve(TAG_PROD_HASH_SIZE);
        for (int64_t i = 0; i < TAG_PROD_HASH_SIZE; i++) {
            cor_tag_prod2idx_[i] = -1;
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
            e.count = 1;

            word2idx_[h] = word_size_++;
            idx2words_.push_back(e);
        } else {
            idx2words_[word2idx_[h]].count++;
        }
        addCorPair(word,cur_prod,1);
        for (uint32_t i = 0; i < cur_tags.size(); ++i) {
            addCorPair(word,cur_tags[i],2);
        }
    }

    void data::addTag(const std::string &tag) {
        int64_t h = getTagHash(tag);
        if (tag2idx_[h] == -1) {
            entry_tag e;
            e.tag = tag;
            e.count = 1;
            e.word_count = 0;

            tag2idx_[h] = tag_size_++;
            idx2tag_.push_back(e);
        }else{
            idx2tag_[tag2idx_[h]].count++;
        }
        addCorPair(tag, cur_prod,3);
    }

    void data::addProd(const std::string &prod) {
        int64_t h = getProdHash(prod);
        if (prod2idx_[h] == -1) {
            entry_prod e;
            e.prod = prod;
            e.count = 1;
            e.word_count = 0;

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

    int64_t data::getTagHash(const std::string &tag) const {
        int64_t h = hash(tag) % TAG_HASH_SIZE;
        while (word2idx_[h] != -1 && idx2tag_[tag2idx_[h]].tag != tag) {
            h = (h + 1) % TAG_HASH_SIZE;
        }
        return h;
    }

    int64_t data::getHashCorPair(const std::string &pair1, const std::string &pair2, uint8_t mode) const {
        std::string key = pair1 + "_" + pair2;
        return getHashCorPair(key, mode);
    }

    int64_t data::getHashCorPair(const std::string &key, uint8_t mode) const {
        int64_t h = hash(key);
        if(mode == 1){
            h = h % WORD_PROD_HASH_SIZE;
            while (cor_word_prod2idx_[h] != -1 && idx2cor_word_prod_[cor_word_prod2idx_[h]] != key) {
                h = (h + 1) % WORD_PROD_HASH_SIZE;
            }
        }else if(mode == 2){
            h = h % WORD_TAG_HASH_SIZE;
            while (cor_word_tag2idx_[h] != -1 && idx2cor_word_tag_[cor_word_tag2idx_[h]] != key) {
                h = (h + 1) % WORD_TAG_HASH_SIZE;
            }
        }else if(mode == 3){
            h = h % TAG_PROD_HASH_SIZE;
            while (cor_tag_prod2idx_[h] != -1 && idx2cor_tag_prod_[cor_tag_prod2idx_[h]] != key) {
                h = (h + 1) % TAG_PROD_HASH_SIZE;
            }
        }
        return h;
    }

    bool data::checkCorPair(const std::string &pair1, const std::string &pair2, uint8_t mode) const {
        int64_t h = getHashCorPair(pair1, pair2, mode);
        if(mode == 1){
            h = h % WORD_PROD_HASH_SIZE;
        }else if(mode == 2){
            h = h % WORD_TAG_HASH_SIZE;
        }else if(mode == 3){
            h = h % TAG_PROD_HASH_SIZE;
        }
        return h != -1;
    }

    void data::addCorPair(const std::string &pair1, const std::string &pair2, uint8_t mode) {
        int h = getHashCorPair(pair1, pair2, mode);
        std::string key = pair1 + "_" + pair2;
        if(mode == 1){
            if(cor_word_prod2idx_[h] == -1){
                cor_word_prod2idx_[h] = word_prod_size_++;
                idx2cor_word_prod_.push_back(key);
            }
        }else if(mode == 2){
            if(cor_word_tag2idx_[h] == -1){
                cor_word_tag2idx_[h] = word_tag_size_++;
                idx2cor_word_tag_.push_back(key);
            }
        }else if(mode == 3){
            if(cor_tag_prod2idx_[h] == -1){
                cor_tag_prod2idx_[h] = tag_prod_size_++;
                idx2cor_tag_prod_.push_back(key);
            }
        }

    }

    int64_t data::getWordId(const std::string &word) const {
        int64_t h = getWordHash(word);
        return word2idx_[h];
    }

    int64_t data::getProdId(const std::string &prod) const {
        int64_t h = getProdHash(prod);
        return prod2idx_[h];
    }

    int64_t data::getTagId(const std::string &tag) const {
        int64_t h = getTagHash(tag);
        return tag2idx_[h];
    }

    std::string data::getWord(uint32_t i) const {
        return idx2words_[i].word;
    }

    void data::readFromFile(std::istream &in) {
        std::string word;
        char c;
        uint8_t cur_mode = 0; //0:prod 1:tag 2:text // 0:user 1:prod 2:rating 3:text
        std::streambuf& sb = *in.rdbuf();
        while ((c = sb.sbumpc()) != EOF) {
            if (c == ' ' || c == '\t' || c == '\v' || c == '\n') {
                if(!word.empty()){
                    if(cur_mode == 0){
                        addProd(word);
                        cur_prod = word;
                    }else if(cur_mode == 1){
                        addTag(word);
                        cur_tags.push_back(word);
                    }else if(cur_mode == 2){
                        addWord(word);
                    }
                }
                if(c == '\t'){
                    cur_mode++;
                }else if(c == '\v'){

                }else if(c == '\n'){
                    cur_mode = 0;
                    cur_prod = UNK;
                    cur_tags.clear();
                }
                word.clear();
            }else{
                word.push_back(c);
            }
        }
        addWord(word); //for last word
        threshold(args_->minCount);
        if (args_->verbose > 0) {
            std::cout << "Number of words:  " << word_size_ << std::endl;
            std::cout << "Number of prods:  " << prod_size_ << std::endl;
            std::cout << "Number of tags:  " << tag_size_ << std::endl;
            std::cout << "Number of word-prod:  " << word_prod_size_ << std::endl;
            std::cout << "Number of word-tag:  " << word_tag_size_ << std::endl;
            std::cout << "Number of tag-prod:  " << tag_prod_size_ << std::endl;
        }
    }

    uint32_t data::getLine(std::istream &in, std::vector<int64_t> &words, std::vector<int64_t> &prods, std::vector<int64_t> &tags,
                           std::minstd_rand &rng) const {
        std::uniform_real_distribution<> uniform(0, 1);
        std::string token;
        uint32_t ntokens = 0;
        words.clear();
        prods.clear();
        tags.clear();
        uint8_t cur_mode = 0;
        std::string word;
        char c;
        std::streambuf& sb = *in.rdbuf();
        while ((c = sb.sbumpc()) != EOF) {
            if (c == ' ' || c == '\t' || c == '\v' || c == '\n') {
                if(!word.empty()){
                    if(cur_mode == 0){
                        int64_t pid = getProdId(word);
                        prods.push_back(pid);
                    }else if(cur_mode == 1){
                        int64_t tid = getTagId(word);
                        tags.push_back(tid);
                    }else if(cur_mode == 2){
                        int64_t wid = getWordId(word);
                        words.push_back(wid);
                        ntokens++;
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
        out.write((char*) &tag_size_, sizeof(uint64_t));
        out.write((char*) &word_prod_size_, sizeof(uint64_t));
        out.write((char*) &word_tag_size_, sizeof(uint64_t));
        out.write((char*) &tag_prod_size_, sizeof(uint64_t));

        for (int64_t i = 0; i < word_size_; i++) {
            entry_word e = idx2words_[i];
            out.write(e.word.data(), e.word.size() * sizeof(char));
            out.put(0);
            out.write((char*) &(e.count), sizeof(uint32_t));
        }
        for (int64_t i = 0; i < prod_size_; i++) {
            entry_prod e = idx2prod_[i];
            out.write(e.prod.data(), e.prod.size() * sizeof(char));
            out.put(0);
            out.write((char*) &(e.count), sizeof(uint32_t));
            out.write((char*) &(e.word_count), sizeof(uint32_t));
        }
        for (int64_t i = 0; i < tag_size_; ++i) {
            entry_tag e = idx2tag_[i];
            out.write(e.tag.data(), e.tag.size() * sizeof(char));
            out.put(0);
            out.write((char*) &(e.count), sizeof(uint32_t));
            out.write((char*) &(e.word_count), sizeof(uint32_t));
        }
        for (int64_t i = 0; i < word_prod_size_; ++i) {
            std::string key = idx2cor_word_prod_[i];
            out.write(key.data(), key.size() * sizeof(char));
            out.put(0);
        }
        for (int64_t i = 0; i < word_tag_size_; ++i) {
            std::string key = idx2cor_word_tag_[i];
            out.write(key.data(), key.size() * sizeof(char));
            out.put(0);
        }
        for (int64_t i = 0; i < tag_prod_size_; ++i) {
            std::string key = idx2cor_tag_prod_[i];
            out.write(key.data(), key.size() * sizeof(char));
            out.put(0);
        }
    }

    void data::load(std::istream &in) {
        word_size_ = 0;
        prod_size_ = 0;
        tag_size_ = 0;
        idx2words_.clear();
        idx2prod_.clear();
        idx2tag_.clear();
        idx2cor_word_prod_.clear();
        idx2cor_word_tag_.clear();
        idx2cor_tag_prod_.clear();

        word2idx_.resize(VOCAB_HASH_SIZE);
        for (int64_t i = 0; i < VOCAB_HASH_SIZE; i++) {
            word2idx_[i] = -1;
        }
        prod2idx_.reserve(PROD_HASH_SIZE);
        for (int64_t i = 0; i < PROD_HASH_SIZE; i++) {
            prod2idx_[i] = -1;
        }
        tag2idx_.reserve(TAG_HASH_SIZE);
        for (int64_t i = 0; i < TAG_HASH_SIZE; i++) {
            tag2idx_[i] = -1;
        }

        cor_word_prod2idx_.reserve(WORD_PROD_HASH_SIZE);
        for (int64_t i = 0; i < WORD_PROD_HASH_SIZE; i++) {
            cor_word_prod2idx_[i] = -1;
        }
        cor_word_tag2idx_.reserve(WORD_TAG_HASH_SIZE);
        for (int64_t i = 0; i < WORD_TAG_HASH_SIZE; i++) {
            cor_word_tag2idx_[i] = -1;
        }
        cor_tag_prod2idx_.reserve(TAG_PROD_HASH_SIZE);
        for (int64_t i = 0; i < TAG_PROD_HASH_SIZE; i++) {
            cor_tag_prod2idx_[i] = -1;
        }


        in.read((char*) &word_size_, sizeof(uint64_t));
        in.read((char*) &prod_size_, sizeof(uint64_t));
        in.read((char*) &tag_size_, sizeof(uint64_t));
        in.read((char*) &word_prod_size_, sizeof(uint64_t));
        in.read((char*) &word_tag_size_, sizeof(uint64_t));
        in.read((char*) &tag_prod_size_, sizeof(uint64_t));

        char c;
        for (int64_t i = 0; i < word_size_; i++) {

            entry_word e;
            while ((c = in.get()) != 0) {
                e.word.push_back(c);
            }
            in.read((char*) &e.count, sizeof(uint32_t));
            idx2words_.push_back(e);
            word2idx_[getWordHash(e.word)] = i;
        }

        for (int64_t i = 0; i < prod_size_; i++) {
            entry_prod e;
            while ((c = in.get()) != 0) {
                e.prod.push_back(c);
            }

            in.read((char*) &e.count, sizeof(uint32_t));
            in.read((char*) &e.word_count, sizeof(uint32_t));

            idx2prod_.push_back(e);
            prod2idx_[getProdHash(e.prod)] = i;
        }

        for (int64_t i = 0; i < tag_size_; i++) {
            entry_tag e;
            while ((c = in.get()) != 0) {
                e.tag.push_back(c);
            }

            in.read((char*) &e.count, sizeof(uint32_t));
            in.read((char*) &e.word_count, sizeof(uint32_t));

            idx2tag_.push_back(e);
            tag2idx_[getTagHash(e.tag)] = i;
        }


        for (int64_t i = 0; i < word_prod_size_; i++) {
            std::string pair;
            while ((c = in.get()) != 0) {
                pair.push_back(c);
            }

            idx2cor_word_prod_.push_back(pair);
            cor_word_prod2idx_[getHashCorPair(pair, 1)] = i;
        }
        for (int64_t i = 0; i < word_tag_size_; i++) {
            std::string pair;
            while ((c = in.get()) != 0) {
                pair.push_back(c);
            }

            idx2cor_word_tag_.push_back(pair);
            cor_word_tag2idx_[getHashCorPair(pair, 2)] = i;
        }
        for (int64_t i = 0; i < tag_prod_size_; i++) {
            std::string pair;
            while ((c = in.get()) != 0) {
                pair.push_back(c);
            }

            idx2cor_tag_prod_.push_back(pair);
            cor_tag_prod2idx_[getHashCorPair(pair, 3)] = i;
        }

    }
}
