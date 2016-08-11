//
// Created by Sanqiang Zhao on 8/10/16.
//
//#include "util.c"
//#include "config.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct prod* idx2prod;

struct prod{
    char* prod_str;
    unsigned int cnt;
};

void populate_prod(){
    idx2prod = (struct prod *)malloc ((size_t) ((n_prod + 1) * sizeof(struct prod)));
    FILE *fin;
    fin = fopen(prod_file, "rb");
    char ch, prodstr[MAX_STRING], prodidx[MAX_STRING];;
    unsigned int word_idx = 0, prod_size = 0, mode = 0; //0 refer word mode and 1 refer to integer mode
    while (!feof(fin)) {
        ch = fgetc(fin);
        if(ch == '\n'){
            prodstr[word_idx] = '\0';
            idx2prod[prod_size].prod_str = (char *)calloc(word_idx, sizeof(char));
            strcpy(idx2prod[prod_size].prod_str,prodstr);
            idx2prod[prod_size].cnt = 0;
            prod_size++;
            word_idx = 0;
        }
        else{
            prodstr[word_idx] = ch;
            word_idx++;
        }
    }
}

void init_prod(){
    populate_prod();
}