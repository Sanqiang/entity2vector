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

struct pair* pairs;

struct pair{
    unsigned int prod;
    unsigned int token;
};

void populate_pair(){
    pairs = (struct pair *)malloc ((size_t) ((n_pair + 1) * sizeof(struct pair)));

    FILE *fin;
    fin = fopen(pair_file, "rb");
    char ch, word[MAX_STRING];
    unsigned int word_idx = 0, prod_indx, token_idx, hash, prod_size = 0, pair_size = 0;

    while (!feof(fin)) {
        ch = fgetc(fin);
        if(ch == '\n'){
            word[word_idx] = '\0';
            token_idx = atoi(word);
            idx2word[token_idx].prods[prod_indx] = 1;
            word_idx = 0;

            pairs[pair_size].prod = prod_indx;
            pairs[pair_size].token = token_idx;
            pair_size++;

        }else if(ch == ' '){
            word[word_idx] = '\0';
            prod_indx = atoi(word);
            idx2prod[prod_indx].cnt++;

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
    populate_pair();
    init_negative_sampling_table();
}

