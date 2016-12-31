//
// Created by Sanqiang Zhao on 8/10/16.
//
//#include "util.c"
//#include "config.c"
//#include "word.c"
//#include "prod.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct pair* idx2pairs;
unsigned int * pairs2idx;

struct pair{
    unsigned long prod;
    unsigned long token;
};

int get_pair_hash(unsigned long token_idx, unsigned long prod_idx){
    unsigned int hash = 0;
    while (token_idx > 0){
        token_idx /= 10;
        hash = hash * 257 + token_idx % 10;
    }
    while (prod_idx > 0){
        prod_idx /= 10;
        hash = hash * 257 + prod_idx % 10;
    }
    return hash % vocab_hash_size;
}

void save_pair_hash(unsigned long token_idx, unsigned long prod_idx, unsigned long pair_size){
    unsigned int hash = get_pair_hash(token_idx, prod_idx);
    while (pairs2idx[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    pairs2idx[hash] = pair_size;
}

int check_pair(unsigned long token_idx, unsigned long prod_idx){
    unsigned int hash = get_pair_hash(token_idx, prod_idx);
    while (1) {
        if (pairs2idx[hash] == -1) return -1;
        //if (!strcmp(key, idx2pairs[pairs2idx[hash]].key)) return pairs2idx[hash];
        if(idx2pairs[pairs2idx[hash]].prod == prod_idx && idx2pairs[pairs2idx[hash]].token == token_idx){
            return pairs2idx[hash];
        }
        hash = (hash + 1) % vocab_hash_size;
    }
}

void populate_pair(){
    idx2pairs = (struct pair *)malloc ((size_t) ((n_pair + 1) * sizeof(struct pair)));

    FILE *fin;
    fin = fopen(pair_file, "rb");
    char ch, word[MAX_STRING];
    unsigned long word_idx = 0, prod_idx, token_idx, hash, prod_size = 0, pair_size = 0;

    while (!feof(fin)) {
        ch = fgetc(fin);
        if(ch == '\n'){
            word[word_idx] = '\0';
            token_idx = atoi(word);
            idx2word[token_idx].cnt++;
            idx2word[token_idx].prods[prod_idx] = true;

            //save_pair_hash(token_idx, prod_idx, pair_size);

            word_idx = 0;

            idx2pairs[pair_size].prod = prod_idx;
            idx2pairs[pair_size].token = token_idx;
            pair_size++;
            /*if(pair_size % 10 == 0){
                printf("%cProgress on populate pairs: %.2f%%  ", 13,  (pair_size) / (float)(n_pair + 1) * 100);
                fflush(stdout);
            }*/


        }else if(ch == ' '){
            word[word_idx] = '\0';
            prod_idx = atoi(word);
            idx2prod[prod_idx].cnt++;

            word_idx = 0;
        }else{
            word[word_idx] = ch;
            word_idx++;
        }
    }
}


void init_negative_sampling_table() {
    long long train_words_pow = 0, a, i;
    float d1, power = 0.75;
    negative_sampling_table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < n_prod; a++) train_words_pow += pow(idx2prod[a].cnt, power);
    i = 0;
    d1 = pow(idx2prod[i].cnt, power) / (float)train_words_pow;
    for (a = 0; a < table_size; a++) {
        negative_sampling_table[a] = i;
        if (a / (float)table_size > d1) {
            i++;
            d1 += pow(idx2prod[i].cnt, power) / (float)train_words_pow;
        }
        if (i >= n_prod) i = n_prod - 1;
    }
}

void init_pair(){
    pairs2idx = (int *)calloc(vocab_hash_size, sizeof(int));
    unsigned long long i;
    for (i = 0; i < vocab_hash_size; i++) pairs2idx[i] = -1;

    populate_pair();

    init_negative_sampling_table();
}

