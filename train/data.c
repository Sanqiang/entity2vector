#include <stdio.h>
#include <stdlib.h>
#include <json.h>
#include "util/config.c"
#include "util/helper.c"

void read_word(FILE *fin, char *word){
    unsigned int ch, word_idx = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if(ch == ' ' || ch == '\t'|| ch == '\v' || ch == '\n'){
            if(word_idx > 0){
                ungetc(ch, fin);
                break;
            }
        }else{
            word[word_idx] = ch;
            word_idx++;
        }
    }
    word[word_idx] = '\0';
}

void read_text(FILE *fin){
    char ch, word[MAX_STRING];
    unsigned int word_idx = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if(ch == '\t'){
            ungetc(ch, fin);
            break;
        }else{
            read_word(fin, word);

        }
    }
}

void read_file(){
    FILE *fin;
    fin = fopen(path_review, "rb");
    char ch, word[MAX_STRING];
    unsigned short mode = 0; //0 indicates text, 1 indicate user_id,2 indicates business_id, 3 indicates stars,

    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == '\n') {
            mode = 0;

        }else if(ch == '\t'){
            if(mode == 0){
                read_text(fin);
            }
            mode += 1;

        }
    }

}

int main(){
    read_file();
}