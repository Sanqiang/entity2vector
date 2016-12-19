/*
 * Word2vec impl
 * */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sys/mman.h>
#include "model.c"

#define MAX_STRING 100//string 类型的最大长度
#define EXP_TABLE_SIZE 1000//这里是用来求sigmoid函数,使用的是一种近似的求法，
#define MAX_EXP 6//只要求球区间为６的即可
#define MAX_SENTENCE_LENGTH 1000//句子最大长度,及包含词数
#define MAX_CODE_LENGTH 40

float *syn0, *syn1, *syn1neg, *expTable;
unsigned short layer1_size, c, n_negative;
unsigned long long vocab_size, cur_data_size, n_dataset;
float alpha;
short  n_threads, iters;
char train_file[MAX_STRING], word_file[MAX_STRING], output_file[MAX_STRING], model_file[MAX_STRING];
const int table_size = 1e8;
int *table;
struct vocab_word *vocab;//词动态数组
struct train_pair *dataset;
int hs = 0;

struct vocab_word {
    unsigned long cn;
    char *word;
    int *point, *code, codelen;//huffman编码对应内节点的路劲
};

char* concat(char *s1, char *s2)
{
    char *result = malloc((strlen(s1)+strlen(s2)+1)* sizeof(char));//+1 for the zero-terminator
    //in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}


void CreateBinaryTree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
    for (a = 0; a < vocab_size; a++)
        count[a] = vocab[a].cn;
    for (a = vocab_size; a < vocab_size * 2; a++)
        count[a] = 1e15;
    pos1 = vocab_size - 1;
    pos2 = vocab_size;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    //count 数组是从大到小排序
    for (a = 0; a < vocab_size - 1; a++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            } else {
                min1i = pos2;
                pos2++;
            }
        } else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            } else {
                min2i = pos2;
                pos2++;
            }
        } else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_size + a] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_size + a;
        parent_node[min2i] = vocab_size + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < vocab_size; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == vocab_size * 2 - 2) break;//到达根节点
        }
        vocab[a].codelen = i;
        vocab[a].point[0] = vocab_size - 2;
        for (b = 0; b < i; b++) {
            vocab[a].code[i - b - 1] = code[b]; //direction from top to down
            vocab[a].point[i - b] = point[b] - vocab_size;//point position from top to down
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

struct train_pair{
    unsigned short context;
    unsigned short target;
};

void populate_vocab(){
    FILE *fin;
    fin = fopen(word_file, "rb");
    vocab = (struct vocab_word *)realloc(vocab, (size_t) ((vocab_size + 1) * sizeof(struct vocab_word)));
    // vocab = (struct vocab_word *)malloc ((size_t) ((vocab_size + 1) * sizeof(struct vocab_word)));
    char word[MAX_STRING];
    int word_idx = 0;
    int num = 0;
    boolean word_mode = 1;

    long long cur_vocab_size = 0;
    char ch;

    while (!feof(fin)) {
        ch = fgetc(fin);

        if(ch == ' '){
            word_mode = 0;
        }else if(ch == '\n'){
            word_mode = 1;
            word[word_idx] = '\0';
            vocab[cur_vocab_size].word = (char *)calloc(strlen(word)+1, sizeof(char));
            strcpy(vocab[cur_vocab_size].word,word);
            vocab[cur_vocab_size].cn = num;
            cur_vocab_size++;
            if (cur_vocab_size >= vocab_size){
                break;
            }
            //fresh var
            word_idx = 0;
            num = 0;

        }else{
            if(word_mode){
                word[word_idx] = ch;
                if(word_idx < MAX_STRING){
                    word_idx ++;
                }

            }else{
                num = num * 10;
                num += ch - '0';
            }
        }
    }
    fclose(fin);
    for(cur_vocab_size=0;cur_vocab_size<vocab_size;cur_vocab_size++){
        vocab[cur_vocab_size].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[cur_vocab_size].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }

}

void init_unigram_table() {
    long long train_words_pow = 0, a, i;
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
    char ch;
    unsigned int num = 0; cur_data_size = 0;

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
    fclose(fin);
}

void init(){
    populate_vocab();
    read_data();
    init_unigram_table();
    CreateBinaryTree();

    //populate exp precomputing table
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    long long i;
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

    //init net
    long long a, b;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn0 == NULL) {printf("syn0 Memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1neg == NULL) {printf("syn1neg Memory allocation failed\n"); exit(1);}
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

    if (hs) {
        a = posix_memalign((void **) &syn1, 128, (long long) vocab_size * layer1_size * sizeof(float));
        if (syn1 == NULL) {
            printf("syn1 Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < vocab_size; a++) {
            for (b = 0; b < layer1_size; b++) {
                syn1[a * layer1_size + b] = 0;
            }
        }
    }
}

void train_thread(void *id) {
    unsigned long long next_random = (long long)id;
    long long thread_id = (long long)id;

    long long pos_st = n_dataset / n_threads * thread_id;
    long long pos_ed = n_dataset / n_threads * (1 + thread_id);

    long long pos = pos_st, last_pos=0;
    int label;
    unsigned int context,target;
    //printf("current thread start! start pos %llu and end pos %llu. \n", pos_st, pos_ed);
    float *neu1 = (float *)calloc(layer1_size, sizeof(float)); // 隐层节点
    float *neu1e = (float *)calloc(layer1_size, sizeof(float)); // 误差累计项，其实对应的是Gneu1
    unsigned  long long l1, l2, c;
    long long a, b, d, i;
    float f, g;

    while (1){

        if(pos - last_pos >= 10000){
            last_pos = pos;
            printf("%cProgress: %.2f%%  ", 13,  (n_threads * (pos - pos_st)) / (float)(n_dataset + 1) * 100);
            fflush(stdout);
        }

        struct train_pair pair = dataset[pos]; //to update
        context = pair.context;
        target = pair.target;
        pos++;

        l1 = context * layer1_size; //location in the hidden layer, update him rather than word

        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

        if (hs){
            for (d = 0; d < vocab[context].codelen; d++) {
                f = 0;
                l2 = vocab[context].point[d] * layer1_size;
                // Propagate hidden -> output
                for (c = 0; c < layer1_size; c++)
                    f += neu1[c] * syn1[c + l2];
                if (f <= -MAX_EXP) continue;
                else if (f >= MAX_EXP) continue;
                else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                // 'g' is the gradient multiplied by the learning rate
                // g 是梯度乘以学习速率
                g = (1 - vocab[context].code[d] - f) * alpha;
                // Propagate errors output -> hidden
                // 累计误差率
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                // Learn weights hidden -> output
                // 更新参数权重值
                for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
            }
        }

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
            }

            l2 = target * layer1_size;

            f = 0;
            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
            if (f > MAX_EXP) {
                g = (label - 1) * alpha;
            }
            else if (f < -MAX_EXP) {
                g = (label - 0) * alpha;
            }
            else {
                g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            }


            for (c = 0; c < layer1_size; c++) {
                neu1e[c] += g * syn1neg[c + l2];
            }
            for (c = 0; c < layer1_size; c++) {
                syn1neg[c + l2] += g * syn0[c + l1];
            }
        }
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];

        if(pos >= pos_ed){
            //printf("finished one loop for one thread %llu. \n", thread_id);
            break;
        }
    }
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void conclude(int ind){
    char str[15];
    sprintf(str, "%d", ind);
    char * path = concat(output_file, str);
    printf("Path: %s \n", path);
    FILE *fo;
    fo = fopen(path, "wb");
    long long a, b;
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (0) for (b = 0; b < layer1_size; b++)
                fwrite(&syn0[a * layer1_size + b], sizeof(float), 1, fo);
        else for (b = 0; b < layer1_size; b++)
                fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void train(){
    pthread_t *pt = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
    int i = 15, p;
    for(; i < 2147400000; i++){
        for (p = 0; p < n_threads; p++) pthread_create(&pt[p], NULL, train_thread, (void *)p);
        for (p = 0; p < n_threads; p++) pthread_join(pt[p], NULL);
        conclude(i);

        //save model
        char str[15];
        char * path;
        sprintf(str, "%d", i);
        path = concat(concat(model_file, str),"_syn0");
        save_model(path, vocab_size, layer1_size,syn0);
        path = concat(concat(model_file, str),"_syn1neg");
        save_model(path, vocab_size, layer1_size,syn1neg);
    }

}

int main(int argc, char **argv) {
    alpha = 0.025;
    vocab_size = 63258;//86632 for rest 123587 for all
    layer1_size = 200;
    n_dataset =  69242128;//717660641 for rest 1203551737 for all
    n_threads = 5;
    n_negative = 10;
//    strcpy(train_file, "/home/sanqiang/git/entity2vector/yelp_rest_allalphaword_yelp_mincnt10_win10/pair.txt");
//    strcpy(word_file, "/home/sanqiang/git/entity2vector/yelp_rest_allalphaword_yelp_mincnt10_win10/pairword.txt");
//    strcpy(output_file, "/home/sanqiang/git/entity2vector/yelp_rest_allalphaword_yelp_mincnt10_win10/result/d200_neg10_");
//    strcpy(model_file, "/home/sanqiang/git/entity2vector/yelp_rest_allalphaword_yelp_mincnt10_win10/model/");
    strcpy(train_file, "/Users/zhaosanqiang916/git/entity2vector/amz_video/pair.txt");
    strcpy(word_file, "/Users/zhaosanqiang916/git/entity2vector/amz_video/pairword.txt");
    strcpy(word_file, "/Users/zhaosanqiang916/git/entity2vector/amz_video/pairword.txt");
    strcpy(word_file, "/Users/zhaosanqiang916/git/entity2vector/amz_video/pairword.txt");
    init();
    printf("finished init for %s.", train_file);
    train();
    printf("finished train for %s .", train_file);

    exit(0);
}
