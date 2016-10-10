//#include <stdio.h>
//#include <stdlib.h>
//#include <sparsehash/dense_hash_map>
//#include "util/helper.c"
//#include "util/config.c"
//
//
//using google::dense_hash_map;
//using std::hash;
//
//dense_hash_map<const char*, int, hash<const char*>, eqstr> word2idx;
//
//bool lookup(dense_hash_map<const char*, int, hash<const char*>, eqstr> dict, char* word){
//    dense_hash_map<const char*, int, hash<const char*>, eqstr>::const_iterator it = dict.find(word);
//    return (it != dict.end() ? true : false);
//}
//
//void read_word(FILE *fin, char *wordx){
//    //read word
//    char word[MAX_STRING];
//    unsigned int ch, word_idx = 0;
//    while (!feof(fin)) {
//        ch = fgetc(fin);
//        if(ch == ' ' || ch == '\t'|| ch == '\v' || ch == '\n'){
//            if(word_idx > 0){
//                ungetc(ch, fin);
//                break;
//            }
//        }else{
//            word[word_idx] = ch;
//            word_idx++;
//        }
//    }
//    word[word_idx] = '\0';
//    //process word
//    //if(!lookup(word2idx, word)){
//        word2idx[word] = word2idx.size();
//        printf("%p \n", &word);
//    //}
//
//}
//
//void read_text(FILE *fin){
//    char ch;
//
//    unsigned int word_idx = 0;
//    while (!feof(fin)) {
//        ch = fgetc(fin);
//        if(ch == '\t'){ //finish sentence
//            ungetc(ch, fin);
//            break;
//        }else{
//            char word[MAX_STRING];
//            read_word(fin, word);
//
//        }
//    }
//}
//
//void read_file(){
//    FILE *fin;
//    fin = fopen(path_review, "rb");
//    char ch, word[MAX_STRING];
//    unsigned short mode = 0; //0 indicates text, 1 indicate user_id,2 indicates business_id, 3 indicates stars,
//
//    while (!feof(fin)) {
//        ch = fgetc(fin);
//        if (ch == '\n') {
//            mode = 0;
//
//        }else if(ch == '\t'){
//            mode += 1;
//        }else{
//            if(mode == 0){
//                read_text(fin);
//            }else if(mode == 1){
//
//            }else if(mode == 2){
//
//            }else if(mode == 3){
//
//            }
//
//        }
//    }
//
//}
//
//void init(){
//    word2idx.set_empty_key(NULL);
//}
//
//int main(){
//    init();
//    read_file();
//
//    printf("done\n");
//    word2idx["x1"]=2;
//    word2idx["x2"]=2;
//    word2idx["x3"]=2;
//    printf("%d", word2idx.size());
//}