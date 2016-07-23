//#include "main.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LENGTH 40

float *entity_vector, *neg_entity_vector, *huffmantree_vector, *expTable;
unsigned int *trained_vector2idx, *prod2idx, vector_length = 3, n_prod = 0, n_pair = 0,
        layer1_size = 200, *negative_sampling_table, n_thread = 5, max_iter = 5, n_negative = 5;
char vector_file[MAX_STRING] = "/Users/zhaosanqiang916/data/sample_vector", pair_file[MAX_STRING] = "";
long long trained_size = 3;
struct trained_vector* idx2trained_vector;
struct pair* pairs;
struct prod* idx2prod;
int hs = 0;
float alpha = 0.025;

const int table_size = 1e8;
const int vocab_hash_size = 30000000;

struct trained_vector{
    char* word;
    float* vector;
};

struct prod{
    char* prod_str;
    unsigned int cnt;
    int *point, *code, codelen;
};

struct pair{
    unsigned int prod;
    unsigned int token;
};

int get_hash(char* word){
    unsigned int i, hash = 0;
    for (i = 0; i < strlen(word); i++) hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;
    return hash;
}

int get_vector_idx(char *word){
    unsigned int hash = get_hash(word);
    while (1) {
        if (trained_vector2idx[hash] == -1) return -1;
        if (!strcmp(word, idx2trained_vector[trained_vector2idx[hash]].word)) return trained_vector2idx[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

void populate_trained_vector(){
    idx2trained_vector = (struct trained_vector *)malloc ((size_t) ((trained_size + 1) * sizeof(struct trained_vector)));

    FILE *fin;
    fin = fopen(vector_file, "rb");
    char ch, word[MAX_STRING], num[MAX_STRING];
    unsigned int word_idx = 0, vector_idx = 0, cur_vector_size = 0, hash;
    unsigned int mode = 0; //0 refer word mode and 1 refer to integer mode 2 refer to decimal mode

    while (!feof(fin)) {
        ch = fgetc(fin);

        if(ch == '\n'){
            word[word_idx] = '\0';
            idx2trained_vector[cur_vector_size].vector[vector_idx] = atof(num);
            vector_idx = 0;
            word_idx = 0;
            cur_vector_size++;
            mode = 0;
        }else if(ch == ' '){
            word[word_idx] = '\0';
            if(mode == 0){
                idx2trained_vector[cur_vector_size].word = (char *)calloc(word_idx, sizeof(char));
                strcpy(idx2trained_vector[cur_vector_size].word,word);

                hash = get_hash(word);
                while (trained_vector2idx[hash] != -1) hash = (hash + 1) % vocab_hash_size;
                trained_vector2idx[hash] = cur_vector_size;

                idx2trained_vector[cur_vector_size].vector = (float *)calloc(vector_length, sizeof(float));
                mode = 1;
            }else if(mode == 1){
                idx2trained_vector[cur_vector_size].vector[vector_idx] = atof(num);
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

int get_prod_idx(char *prod){
    unsigned int hash = get_hash(prod);
    while (1) {
        if (prod2idx[hash] == -1) return -1;
        if (!strcmp(prod, idx2trained_vector[prod2idx[hash]].word)) return prod2idx[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

void populate_pair(){
    pairs = (struct pair *)malloc ((size_t) ((n_pair + 1) * sizeof(struct pair)));
    idx2prod = (struct prod *)malloc ((size_t) ((n_prod + 1) * sizeof(struct prod)));

    FILE *fin;
    fin = fopen(pair_file, "rb");
    char ch, word[MAX_STRING];
    unsigned int word_idx = 0, prod_indx, token_idx, hash, prod_size = 0, pair_size = 0;

    while (!feof(fin)) {
        ch = fgetc(fin);
        if(ch == '\n'){
            word[word_idx] = '\0';
            token_idx = get_vector_idx(word);

            pairs[pair_size].prod = prod_indx;
            pairs[pair_size].token = token_idx;
            pair_size++;

        }else if(ch == ' '){
            word[word_idx] = '\0';
            prod_indx = get_prod_idx(word);
            if(prod_indx == -1){
                hash = get_hash(word);
                while (prod2idx[hash] != -1) hash = (hash + 1) % vocab_hash_size;
                prod2idx[hash] = prod_size;
                idx2prod[prod_size].prod_str = (char *)calloc(word_idx, sizeof(char));
                idx2prod[prod_size].cnt = 0;
                prod_size++;
            }
            prod_indx = get_prod_idx(word);
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

void init_huffman_tree() {
    long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
    char code[MAX_CODE_LENGTH];
    long long *count = (long long *)calloc(n_prod * 2 + 1, sizeof(long long));
    long long *binary = (long long *)calloc(n_prod * 2 + 1, sizeof(long long));
    long long *parent_node = (long long *)calloc(n_prod * 2 + 1, sizeof(long long));
    for (a = 0; a < n_prod; a++)
        count[a] = idx2prod[a].cnt;
    for (a = n_prod; a < n_prod * 2; a++)
        count[a] = 1e15;
    pos1 = n_prod - 1;
    pos2 = n_prod;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    //count 数组是从大到小排序
    for (a = 0; a < n_prod - 1; a++) {
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
        count[n_prod + a] = count[min1i] + count[min2i];
        parent_node[min1i] = n_prod + a;
        parent_node[min2i] = n_prod + a;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (a = 0; a < n_prod; a++) {
        b = a;
        i = 0;
        while (1) {
            code[i] = binary[b];
            point[i] = b;
            i++;
            b = parent_node[b];
            if (b == n_prod * 2 - 2) break;//到达根节点
        }
        idx2prod[a].codelen = i;
        idx2prod[a].point[0] = n_prod - 2;
        for (b = 0; b < i; b++) {
            idx2prod[a].code[i - b - 1] = code[b]; //direction from top to down
            idx2prod[a].point[i - b] = point[b] - n_prod;//point position from top to down
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void init(){
    //init hash table
    trained_vector2idx = (int *)calloc(vocab_hash_size, sizeof(int));
    unsigned int i = 0;
    for (i = 0; i < vocab_hash_size; i++) trained_vector2idx[i] = -1;
    for (i = 0; i < vocab_hash_size; i++) prod2idx[i] = -1;

    populate_trained_vector();
    populate_pair();

    //populate exp precomputing table
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

    init_negative_sampling_table();
    init_huffman_tree();

    //init net
    long long a, b;
    a = posix_memalign((void **)&entity_vector, 128, (long long)n_prod * layer1_size * sizeof(float));
    if (entity_vector == NULL) {printf("entity_vector memory allocation failed\n"); exit(1);}
    a = posix_memalign((void **)&neg_entity_vector, 128, (long long)n_prod * layer1_size * sizeof(float));
    if (neg_entity_vector == NULL) {printf("neg_entity_vector memory allocation failed\n"); exit(1);}
    for (a = 0; a < n_prod; a++){
        for (b = 0; b < layer1_size; b++) {
            neg_entity_vector[a * layer1_size + b] = 0;
        }
    }

    unsigned long long next_random = 1;
    for (a = 0; a < n_prod; a++){
        for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            entity_vector[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
        }
    }

    if (hs) {
        a = posix_memalign((void **) &huffmantree_vector, 128, (long long) n_prod * layer1_size * sizeof(float));
        if (huffmantree_vector == NULL) {
            printf("syn1 Memory allocation failed\n");
            exit(1);
        }
        for (a = 0; a < n_prod; a++) {
            for (b = 0; b < layer1_size; b++) {
                huffmantree_vector[a * layer1_size + b] = 0;
            }
        }
    }
}

void train_thread(void *id) {
    unsigned long long next_random = (long long)id;
    long long thread_id = (long long)id;

    long long pos_st = n_pair / n_thread * thread_id;
    long long pos_ed = n_pair / n_thread * (1 + thread_id);

    long long pos = pos_st, last_pos=0;
    long long i, c, d, l1, l2;
    float f, g;
    int local_iter = max_iter, label;
    printf("current thread start! start pos %llu and end pos %llu. \n", pos_st, pos_ed);
    float *neu1 = (float *)calloc(layer1_size, sizeof(float)); // 隐层节点
    float *neu1e = (float *)calloc(layer1_size, sizeof(float)); // 误差累计项，其实对应的是Gneu1

    while (1){

        if(pos - last_pos >= 10000){
            last_pos = pos;
            printf("%cProgress: %.2f%%  ", 13,  (n_thread * (last_pos+pos)) / (float)(max_iter * n_pair + 1) * 100);
            fflush(stdout);
        }

        struct pair pair = pairs[pos]; //to update
        unsigned short context = pair.prod;
        unsigned short target = pair.prod;
        pos++;

        l1 = context * layer1_size; //location in the hidden layer, update him rather than word

        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

        if (hs){
            for (d = 0; d < idx2prod[context].codelen; d++) {
                f = 0;
                l2 = idx2prod[context].point[d] * layer1_size;
                // Propagate hidden -> output
                for (c = 0; c < layer1_size; c++)
                    f += neu1[c] * huffmantree_vector[c + l2];
                if (f <= -MAX_EXP) continue;
                else if (f >= MAX_EXP) continue;
                else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                // 'g' is the gradient multiplied by the learning rate
                // g 是梯度乘以学习速率
                g = (1 - idx2prod[context].code[d] - f) * alpha;
                // Propagate errors output -> hidden
                // 累计误差率
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * huffmantree_vector[c + l2];
                // Learn weights hidden -> output
                // 更新参数权重值
                for (c = 0; c < layer1_size; c++) huffmantree_vector[c + l2] += g * neu1[c];
            }
        }

        for (i = 0; i <= (n_negative+1); ++i) {
            if(i == 0){
                label = 1;
            }
            else{
                label = 0;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = negative_sampling_table[(next_random >> 16) % table_size];
                if (target == 0) target = next_random % (n_prod - 1) + 1;
                if (target == context) continue;
                target = 0;
            }

            l2 = target * layer1_size;

            f = 0;
            for (c = 0; c < layer1_size; c++) f += entity_vector[c + l1] * neg_entity_vector[c + l2];
            if (f > MAX_EXP) g = (label - 1) * alpha;else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            for (c = 0; c < layer1_size; c++) neu1e[c] += g * neg_entity_vector[c + l2];
            for (c = 0; c < layer1_size; c++) neg_entity_vector[c + l2] += g * entity_vector[c + l1];
        }
        for (c = 0; c < layer1_size; c++) entity_vector[c + l1] += neu1e[c];

        pos += 1;
        if(pos >= pos_ed){
            printf("finished one loop for one thread %llu. \n", thread_id);
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
    pthread_t *pt = (pthread_t *)malloc(n_thread * sizeof(pthread_t));
    for(int i = 0; i < 2147400000; i++){
        for (int p = 0; p < n_thread; p++) pthread_create(&pt[p], NULL, train_thread, (void *)p);
        for (int p = 0; p < n_thread; p++) pthread_join(pt[p], NULL);
        //conclude(i);
    }

}

int main(int argc, char **argv) {
    init();
    exit(0);
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    