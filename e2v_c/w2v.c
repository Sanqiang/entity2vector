#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <jmorecfg.h>
#include <sys/mman.h>

#define MAX_STRING 100//string 类型的最大长度
#define EXP_TABLE_SIZE 1000//这里是用来求sigmoid函数,使用的是一种近似的求法，
#define MAX_EXP 6//只要求球区间为６的即可
#define MAX_SENTENCE_LENGTH 1000//句子最大长度,及包含词数

float *syn0, *syn1, *syn1neg, *expTable;
int layer1_size, c, n_negative;
long long l1, l2, vocab_size, cur_data_size, n_dataset;
int a, b;
long p;
float f, g, alpha;
int  label, i, n_threads, num, iters;
char train_file[MAX_STRING], word_file[MAX_STRING];
char ch;

const int table_size = 1e8;
int *table;
struct vocab_word *vocab;//词动态数组
struct train_pair *dataset;
//struct train_pair dataset2[2453688725];


struct vocab_word {
    long long cn;//词频
    char *word;
};

struct train_pair{
    unsigned short context;
    unsigned short target;
};

void populate_vocab(){
    FILE *fin;
    fin = fopen(word_file, "rb");
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));

    char word[MAX_STRING];
    int word_idx = 0;
    int num = 0;
    boolean word_mode = 1;
    long long cur_vocab_size = 0;

    while (!feof(fin)) {
        ch = fgetc(fin);

        if(ch == ' '){
            word_mode = 0;
        }else if(ch == '\n'){
            word_mode = 1;
            word[word_idx] = 0;
            vocab[cur_vocab_size].word = (char *)calloc(word_idx, sizeof(char));
            strcpy(vocab[cur_vocab_size].word,word);
            vocab[cur_vocab_size].cn = num;
            cur_vocab_size++;
            //fresh var
            word_idx = 0;
            num = 0;

        }else{
            if(word_mode){
                word[word_idx] = ch;
                word_idx ++;
            }else{
                num = num * 10;
                num += ch - '0';
            }
        }
    }
}

void init_unigram_table() {
    long long train_words_pow = 0;
    float d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    i = 0;
    d1 = pow(vocab[i].cn, power) / (float)train_words_pow;
    for (a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (float)table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / (float)train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

void read_data(){
    FILE *fin;
    dataset = (struct train_pair *)malloc ((size_t) ((n_dataset + 1) * sizeof(struct train_pair)));
    fin = fopen(train_file, "rb");

    num = 0; cur_data_size = 0;

    while (!feof(fin)) {
        ch = fgetc(fin);
        if(ch == ' '){
            dataset[cur_data_size].target = num;
            num = 0;
        }else if(ch == '\n'){
            dataset[cur_data_size].context = num;
            cur_data_size++;
            num = 0;
        }else{
            num = num * 10;
            num += ch - '0';
        }
    }

}

void init(){
    read_data();
    populate_vocab();
    init_unigram_table();

    //populate exp precomputing table
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

    //init net
    long long a;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(float));
    for (a = 0; a < vocab_size; a++){
        for (b = 0; b < layer1_size; b++) {
            syn1neg[a * layer1_size + b] = 0;
        }
    }

    unsigned long long next_random = 1;
    for (a = 0; a < vocab_size; a++){
        for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
        }
    }
}

void train_thread(void *id) {
    unsigned long long next_random = (long long)id;

    long long pos_st = n_dataset / n_threads * (long long)id;
    long long pos_ed = n_dataset / n_threads * (1 + (long long)id);

    long long pos = pos_st;
    int local_iter = iters;
    printf("current thread start! start pos %llu and end pos %llu. \n", pos_st, pos_ed);
    float *neu1 = (float *)calloc(layer1_size, sizeof(float)); // 隐层节点
    float *neu1e = (float *)calloc(layer1_size, sizeof(float)); // 误差累计项，其实对应的是Gneu1

    while (1){
        struct train_pair pair = dataset[pos]; //to update
        unsigned short context = pair.context;
        unsigned short target = pair.target;
        pos++;

        l1 = context * layer1_size; //location in the hidden layer, update him rather than word


        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

        for (i = 0; i <= (n_negative+1); ++i) {
            if(i == 0){
                label = 1;
            }
            else{
                label = 0;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (vocab_size - 1) + 1;
                if (target == context) continue;
                target = 0;
            }

            l2 = target * layer1_size;

            f = 0;
            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
            for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];

        pos += 1;
        if(pos <= pos_ed){
            printf("finished one loop for one thread.");
            local_iter--;
            pos = pos_st;
            if(local_iter == 0){
                break;
            }
        }
    }
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void train(){
    pthread_t *pt = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
    for (p = 0; p < n_threads; p++) pthread_create(&pt[p], NULL, train_thread, (void *)p);
    for (p = 0; p < n_threads; p++) pthread_join(pt[p], NULL);
}

void conclude(){
//    for (a = 0; a < vocab_size; a++){
//        for (b = 0; b < layer1_size; b++) {
//            printf("%f, ", syn0[a * layer1_size + b]);
//        }
//        printf("\n");
//    }
    FILE *fo;
    fo = fopen("/home/sanqiang/Documents/git/entity2vector/yelp_ny_pos/syn0.txt", "wb");
    for (a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (0) for (b = 0; b < layer1_size; b++)
                fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo);
        else for (b = 0; b < layer1_size; b++)
                fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }

}

int main(int argc, char **argv) {

    vocab_size = 30003;
    layer1_size = 100;
    n_dataset = 2453688725;
    n_threads = 5;
    n_negative = 5;
    //old setting /Users/zhaosanqiang916/ClionProjects/e2v/
    strcpy(train_file, "/home/sanqiang/Documents/git/entity2vector/yelp_ny_pos/pair.txt");
    strcpy(word_file, "/home/sanqiang/Documents/git/entity2vector/yelp_ny_pos/pairword.txt");
    init();
    printf("finished init.");
    train();
    printf("finished train.");
    conclude();

    exit(0);
}
