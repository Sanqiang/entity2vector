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

        if(args_->neg_flag == 0) {
            idx2cor_word_prod_.clear();
            idx2cor_word_tag_.clear();
            idx2cor_tag_prod_.clear();
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
    }

    uint64_t data::hash(const std::string &str) const {
        uint64_t h = 2166136261;
        for (size_t i = 0; i < str.size(); i++) {
            h = h ^ uint32_t(str[i]);
            h = h * 16777619;
        }
        return h;
    }

    int64_t data::addWord(const std::string &word) {
        if(word == UNK){
            return -1;
        }
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

        if(args_->neg_flag == 0) {
            addCorPair(word, cur_prod, 1);
            for (uint32_t tag_idx = 0; tag_idx < cur_tags.size(); ++tag_idx) {
                addCorPair(word, cur_tags[tag_idx], 2);
            }
        }
        return h;
    }

    int64_t data::addTag(const std::string &tag) {
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
        if(args_->neg_flag == 0) {
            addCorPair(tag, cur_prod,3);
        }

        return h;
    }

    int64_t data::addProd(const std::string &prod) {
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

        return h;
    }

    int64_t data::getProdHash(const std::string &prod) const {
        uint64_t h = hash(prod) % PROD_HASH_SIZE;
        while (prod2idx_[h] != -1 && idx2prod_[prod2idx_[h]].prod != prod) {
            h = (h + 1) % PROD_HASH_SIZE;
        }
        return h;
    }

    int64_t data::getWordHash(const std::string &word) const {
        uint64_t h = hash(word) % VOCAB_HASH_SIZE;
        while (word2idx_[h] != -1 && idx2words_[word2idx_[h]].word != word) {
            h = (h + 1) % VOCAB_HASH_SIZE;
        }
        return h;
    }

    int64_t data::getTagHash(const std::string &tag) const {
        uint64_t h = hash(tag) % TAG_HASH_SIZE;
        while (tag2idx_[h] != -1 && idx2tag_[tag2idx_[h]].tag != tag) {
            h = (h + 1) % TAG_HASH_SIZE;
        }
        return h;
    }

    int64_t data::getHashCorPair(const std::string &pair1, const std::string &pair2, uint8_t mode) const {
        std::string key = pair1 + "_" + pair2;
        return getHashCorPair(key, mode);
    }

    int64_t data::getHashCorPair(const std::string &key, uint8_t mode) const {
        uint64_t h = hash(key);
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
        if(args_->neg_flag == 0) {
            int64_t h = getHashCorPair(pair1, pair2, mode);
            if (mode == 1) {
                return cor_word_prod2idx_[h] != -1;
            } else if (mode == 2) {
                return cor_word_tag2idx_[h] != -1;
            } else if (mode == 3) {
                return cor_tag_prod2idx_[h] != -1;
            }
        }else if(args_->neg_flag == 1){
            if (mode == 1) {
                int64_t word_id = getWordId(pair1);
                int64_t prod_id = getProdId(pair2);
                return checkCorPair(word_id, prod_id, 1);
            } else if (mode == 2) {
                int64_t word_id = getWordId(pair1);
                int64_t tag_id = getTagId(pair2);
                return checkCorPair(word_id, tag_id, 2);
            } else if (mode == 3) {
                int64_t prod_id = getProdId(pair1);
                int64_t tag_id = getTagId(pair2);
                return checkCorPair(prod_id, tag_id, 3);
            }
        }
        return 1;
    }

    bool data::checkCorPair(const int64_t pair1_idx, const int64_t pair2_idx, uint8_t mode) const {
        if(args_->neg_flag == 0) {
            std::string key1 = "", key2 = "";
            if (mode == 1) {
                key1 = getWord(pair1_idx);
                key2 = getProd(pair2_idx);
            } else if (mode == 2) {
                key1 = getWord(pair1_idx);
                key2 = getTag(pair2_idx);
            } else if (mode == 3) {
                key1 = getTag(pair1_idx);
                key2 = getProd(pair2_idx);
            }
            return checkCorPair(key1, key2, mode);
        }else if(args_->neg_flag == 1){
            if (mode == 1) {
                return word_prod_tab[pair1_idx*prod_size_ + pair2_idx];
            } else if (mode == 2) {
                return word_prod_tab[pair1_idx*tag_size_ + pair2_idx];
            } else if (mode == 3) {
                return word_prod_tab[pair1_idx*tag_size_ + pair2_idx];
            }
        }
    }

    void data::addCorPair(const std::string &pair1, const std::string &pair2, uint8_t mode) {
        if(args_->neg_flag == 0) {
            int h = getHashCorPair(pair1, pair2, mode);
            std::string key = pair1 + "_" + pair2;
            if (mode == 1) {
                if (cor_word_prod2idx_[h] == -1) {
                    cor_word_prod2idx_[h] = word_prod_size_++;
                    idx2cor_word_prod_.push_back(key);
                }
            } else if (mode == 2) {
                if (cor_word_tag2idx_[h] == -1) {
                    cor_word_tag2idx_[h] = word_tag_size_++;
                    idx2cor_word_tag_.push_back(key);
                }
            } else if (mode == 3) {
                if (cor_tag_prod2idx_[h] == -1) {
                    cor_tag_prod2idx_[h] = tag_prod_size_++;
                    idx2cor_tag_prod_.push_back(key);
                }
            }
        }else if(args_->neg_flag == 1){
            if (mode == 1) {
                int64_t word_id = getWordId(pair1);
                int64_t prod_id = getProdId(pair2);
                if(word_id >= 0 && !word_prod_tab[word_id * prod_size_ + prod_id]){
                    word_prod_tab[word_id * prod_size_ + prod_id] = true;
                    word_prod_size_++;
                }
            } else if (mode == 2) {
                int64_t word_id = getWordId(pair1);
                int64_t tag_id = getTagId(pair2);
                if(word_id >= 0 && !word_tag_tab[word_id * tag_size_ + tag_id]) {
                    word_tag_tab[word_id * tag_size_ + tag_id] = true;
                    word_tag_size_++;
                }
            } else if (mode == 3) {
                int64_t tag_id = getTagId(pair1);
                int64_t prod_id = getProdId(pair2);
                if(!tag_prod_tab[tag_id * prod_size_ + prod_id]){
                    tag_prod_tab[tag_id * prod_size_ + prod_id] = true;
                    tag_prod_size_++;
                }
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

    std::string data::getWord(int64_t i) const {
        return idx2words_[i].word;
    }

    std::string data::getProd(int64_t i) const {
        return idx2prod_[i].prod;
    }

    std::string data::getTag(int64_t i) const {
        return idx2tag_[i].tag;
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
                        int64_t prod_id = addProd(word);
                        cur_prod = word;
                    }else if(cur_mode == 1){
                        int64_t tag_id = addTag(word);
                        cur_tags.push_back(word);
                    }else if(cur_mode == 2){
                        int64_t word_id = addWord(word);
                    }
                }
                if(c == '\t'){
                    cur_mode++;
                }else if(c == '\v'){

                }else if(c == '\n'){
                    cur_mode = 0;
                    cur_prod = UNK;
                    cur_tags.clear();
                    ++data_size;
                }
                word.clear();
            }else{
                word.push_back(c);
            }
        }
        threshold(args_->minCount);

        if(args_->memory_mode == 1){
            cur_memory_words.clear();
            cur_memory_prods.clear();
            cur_memory_tags.clear();

            in.clear();
            in.seekg(std::streampos(0));

            std::streambuf& sb = *in.rdbuf();
            while ((c = sb.sbumpc()) != EOF) {
                if (c == ' ' || c == '\t' || c == '\v' || c == '\n') {
                    if(!word.empty()){
                        if(cur_mode == 0){
                            int64_t prod_id = getProdId(word);
                            cur_memory_prods.push_back(prod_id);
                        }else if(cur_mode == 1){
                            int64_t tag_id = getTagId(word);
                            cur_memory_tags.push_back(tag_id);
                        }else if(cur_mode == 2){
                            int64_t word_id = getWordId(word);
                            cur_memory_words.push_back(word_id);
                        }
                    }
                    if(c == '\t'){
                        cur_mode++;
                    }else if(c == '\v'){

                    }else if(c == '\n'){
                        data_memory_words.push_back(cur_memory_words);
                        data_memory_prods.push_back(cur_memory_prods);
                        data_memory_tags.push_back(cur_memory_tags);
                        cur_memory_words.clear();
                        cur_memory_prods.clear();
                        cur_memory_tags.clear();
                    }
                    word.clear();
                }else{
                    word.push_back(c);
                }
            }

            uint64_t step = floor(data_size / args_->thread);
            for (int i = 0; i < args_->thread; ++i) {
                pointers.push_back(i*step);
            }
        }


        if(args_->neg_flag == 1){
            word_prod_tab = new bool[word_size_*prod_size_];
            for (uint64_t word_idx = 0; word_idx < word_size_; ++word_idx) {
                for (uint64_t prod_idx = 0; prod_idx < prod_size_; ++prod_idx) {
                    word_prod_tab[word_idx*prod_size_ + prod_idx] = false;
                }
            }
            word_tag_tab = new bool[word_size_*tag_size_];
            for (uint64_t word_idx = 0; word_idx < word_size_; ++word_idx) {
                for (uint64_t tag_idx = 0; tag_idx < tag_size_; ++tag_idx) {
                    word_tag_tab[word_idx * tag_size_ + tag_idx] = false;
                }
            }
            tag_prod_tab = new bool[tag_size_*prod_size_];
            for (uint64_t tag_idx = 0; tag_idx < tag_size_; ++tag_idx) {
                for (uint64_t prod_idx = 0; prod_idx < prod_size_; ++prod_idx) {
                    tag_prod_tab[tag_idx*prod_size_ + prod_idx] = false;
                }
            }

            in.seekg(0, std::ios::beg);
            cur_mode = 0;
            cur_prod = UNK;
            cur_tags.clear();
            std::streambuf& sb = *in.rdbuf();

            while ((c = sb.sbumpc()) != EOF) {
                if (c == ' ' || c == '\t' || c == '\v' || c == '\n') {
                    if(!word.empty()){
                        if(cur_mode == 0){
                            cur_prod = word;
                        }else if(cur_mode == 1){
                            cur_tags.push_back(word);
                        }else if(cur_mode == 2){
                            if(word != UNK) {
                                addCorPair(word, cur_prod, 1);
                                for (uint32_t tag_idx = 0; tag_idx < cur_tags.size(); ++tag_idx) {
                                    addCorPair(word, cur_tags[tag_idx], 2);
                                    addCorPair(cur_tags[tag_idx], cur_prod, 3);
                                }
                            }
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
        }

        if (args_->verbose > 0) {
            std::cout <<  std::endl;
            std::cout << "Number of words:  " << word_size_ << std::endl;
            std::cout << "Number of prods:  " << prod_size_ << std::endl;
            std::cout << "Number of tags:  " << tag_size_ << std::endl;
            std::cout << "Number of word-prod:  " << word_prod_size_ << std::endl;
            std::cout << "Number of word-tag:  " << word_tag_size_ << std::endl;
            std::cout << "Number of tag-prod:  " << tag_prod_size_ << std::endl;
        }
    }

    int32_t data::getLine(std::istream &in, std::vector<int64_t> &words, std::vector<int64_t> &prods, std::vector<int64_t> &tags,
                           std::minstd_rand &rng, uint32_t threadId) {
        if(args_->memory_mode == 1){
            uint64_t position = pointers[threadId];
            words = data_memory_words[position];
            prods = data_memory_prods[position];
            tags = data_memory_tags[position];
            pointers[threadId]++;
            if (position >= data_size){
                pointers[threadId] = 0;
                return -1;
            }else{
                return 1;
            }


        }else if(args_->memory_mode == 0) {

            std::uniform_real_distribution<> uniform(0, 1);
            std::string token;
            uint32_t ntokens = 0;
            words.clear();
            prods.clear();
            tags.clear();
            uint8_t cur_mode = 0;
            std::string word;
            char c;
            std::streambuf &sb = *in.rdbuf();
            while ((c = sb.sbumpc()) != EOF) {
                if (c == ' ' || c == '\t' || c == '\v' || c == '\n') {
                    if (!word.empty()) {
                        if (cur_mode == 0) {
                            int64_t pid = getProdId(word);
                            prods.push_back(pid);
                        } else if (cur_mode == 1) {
                            int64_t tid = getTagId(word);
                            tags.push_back(tid);
                        } else if (cur_mode == 2) {
                            int64_t wid = getWordId(word);
                            words.push_back(wid);
                            ntokens++;
                        }
                    }

                    if (c == '\t') {
                        cur_mode++;
                    } else if (c == '\v') {

                    } else if (c == '\n') {
                        cur_mode = 0;
                        break;
                    }
                    word.clear();
                } else {
                    word.push_back(c);
                }
            }
            if ((c = sb.sbumpc()) == EOF) {
                in.clear();
                in.seekg(std::streampos(0));
                return -1;
            } else {
                sb.sputbackc(c);
            }
            return ntokens;
        }
    }

    uint64_t data::nwords() {
        return word_size_;
    }

    uint64_t data::nprods() {
        return prod_size_;
    }

    uint64_t data::ntags() {
        return tag_size_;
    }

    std::vector<uint32_t> data::getWordCounts() {
        std::vector<uint32_t> counts;
        for (auto& w : idx2words_) {
            counts.push_back(w.count);
        }
        return counts;
    }

    std::vector<uint32_t> data::getProdCounts() {
        std::vector<uint32_t> counts;
        for (auto& w : idx2prod_) {
            counts.push_back(w.count);
        }
        return counts;
    }

    std::vector<uint32_t> data::getTagCounts() {
        std::vector<uint32_t> counts;
        for (auto& w : idx2tag_) {
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
            if(it->word == UNK){
                continue;
            }
            uint32_t h = getWordHash(it->word);
            word2idx_[h] = word_size_++;
        }
    }

    void data::save(std::ostream &out) const {
        out.write((char*) &word_size_, sizeof(uint64_t));
        out.write((char*) &prod_size_, sizeof(uint64_t));
        out.write((char*) &tag_size_, sizeof(uint64_t));

        out.write((char *) &word_prod_size_, sizeof(uint64_t));
        out.write((char *) &word_tag_size_, sizeof(uint64_t));
        out.write((char *) &tag_prod_size_, sizeof(uint64_t));

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
        if(args_->neg_flag == 0) {
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
    }

    //todo data serialize for new neg_flag = 1
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

        if (args_->neg_flag == 0) {
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

        in.read((char *) &word_size_, sizeof(uint64_t));
        in.read((char *) &prod_size_, sizeof(uint64_t));
        in.read((char *) &tag_size_, sizeof(uint64_t));
        in.read((char *) &word_prod_size_, sizeof(uint64_t));
        in.read((char *) &word_tag_size_, sizeof(uint64_t));
        in.read((char *) &tag_prod_size_, sizeof(uint64_t));

        char c;
        for (int64_t i = 0; i < word_size_; i++) {

            entry_word e;
            while ((c = in.get()) != 0) {
                e.word.push_back(c);
            }
            in.read((char *) &e.count, sizeof(uint32_t));
            idx2words_.push_back(e);
            word2idx_[getWordHash(e.word)] = i;
        }

        for (int64_t i = 0; i < prod_size_; i++) {
            entry_prod e;
            while ((c = in.get()) != 0) {
                e.prod.push_back(c);
            }

            in.read((char *) &e.count, sizeof(uint32_t));
            in.read((char *) &e.word_count, sizeof(uint32_t));

            idx2prod_.push_back(e);
            prod2idx_[getProdHash(e.prod)] = i;
        }

        for (int64_t i = 0; i < tag_size_; i++) {
            entry_tag e;
            while ((c = in.get()) != 0) {
                e.tag.push_back(c);
            }

            in.read((char *) &e.count, sizeof(uint32_t));
            in.read((char *) &e.word_count, sizeof(uint32_t));

            idx2tag_.push_back(e);
            tag2idx_[getTagHash(e.tag)] = i;
        }

        if (args_->neg_flag == 0) {
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
}
