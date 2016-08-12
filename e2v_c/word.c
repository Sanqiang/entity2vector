//
// Created by Sanqiang Zhao on 8/10/16.
//
//#include "util.c"
//#include "config.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct word* idx2word;
unsigned int *word2idx;

struct word{
    char* word;
    float* vector;
    int* prods;
};

int get_hash(char* word){
    unsigned int i, hash = 0;
    for (i = 0; i < strlen(word); i++) hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;
    return hash;
}

int get_word_from_idx(char * word){
    unsigned int hash = get_hash(word);
    while (1) {
        if (word2idx[hash] == -1) return -1;
        if (!strcmp(word, idx2word[word2idx[hash]].word)) return word2idx[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
}

struct word get_idx_from_word(unsigned int idx){
    return idx2word[idx];
}

void save_word_hash(char * word, unsigned int cur_vector_size){
    unsigned int hash = get_hash(word);
    while (word2idx[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    word2idx[hash] = cur_vector_size;
}


void populate_wordvector(){
    idx2word = (struct word *)malloc ((size_t) ((word_size + 1) * sizeof(struct word)));

    FILE *fin;
    fin = fopen(vector_file, "rb");
    char ch, word[MAX_STRING], num[MAX_STRING];
    unsigned long long word_idx = 0, vector_idx = 0, cur_vector_size = 0, hash, i;
    unsigned int mode = 0; //0 refer word mode and 1 refer to integer mode

    while (!feof(fin)) {
        ch = fgetc(fin);

        if(ch == '\n'){
            word[word_idx] = '\0';
            idx2word[cur_vector_size].vector[vector_idx] = atof(num);

            vector_idx = 0;
            word_idx = 0;
            cur_vector_size++;
            mode = 0;
        }else if(ch == ' '){
            word[word_idx] = '\0';
            if(mode == 0){
                idx2word[cur_vector_size].prods = (int *)calloc(n_prod, sizeof(int));
                for(i=0;i<n_prod;i++){
                    idx2word[cur_vector_size].prods[i] = 0;
                }
                idx2word[cur_vector_size].vector = (float *)calloc(layer1_size, sizeof(float));

                idx2word[cur_vector_size].word = (char *)calloc(word_idx, sizeof(char));
                strcpy(idx2word[cur_vector_size].word,word);

                save_word_hash(word, cur_vector_size);
                printf("%s \n",word);


                mode = 1;
            }else if(mode == 1){
                idx2word[cur_vector_size].vector[vector_idx] = atof(num);
                vector_idx++;
            }
            word_idx = 0;
        }
        else{
            if(mode == 0){
                word[word_idx] = ch;
                word_idx++;
            }else if(mode == 1){
                num[word_idx] = ch;
                word_idx++;
            }
        }
    }
}

void init_word(){
    //init hash table
    word2idx = (int *)calloc(vocab_hash_size, sizeof(int));
    unsigned long long i;
    for (i = 0; i < vocab_hash_size; i++) word2idx[i] = -1;


    populate_wordvector();
}