#include "config.c"
#include "util.c"
#include "word.c"
#include "prod.c"
#include "pair.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "model.c"

float *entity_vector, *neg_entity_vector;
float alpha = 0.01;

void init(){
    init_util();
    init_word();
    init_prod();
    init_pair();
    //init net
    long long a, b;
    a = posix_memalign((void **)&entity_vector, 128, (long long)(n_prod + word_size + n_tag) * layer1_size * sizeof(float));
    if (entity_vector == NULL) {printf("entity_vector memory allocation failed\n"); exit(1);}

    unsigned long long next_random = 1;
    for (a = 0; a < n_prod + word_size + n_tag; a++){
        for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            entity_vector[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
        }
    }
}

void train_thread(void *id) {
    unsigned long long next_random = (long long)id;
    long long thread_id = (long long)id;

    long long pos_st = n_pair / n_thread * thread_id;
    long long pos_ed = n_pair / n_thread * (1 + thread_id);

    long long pos = pos_st, last_pos=0;
    long long i, c, l1;
    float f, g;
    int label;
    //printf("current thread start! start pos %llu and end pos %llu. \n", pos_st, pos_ed);
    float *neu1 = (float *)calloc(layer1_size, sizeof(float)); // 隐层节点
    float *neu1e = (float *)calloc(layer1_size, sizeof(float)); // 误差累计项，其实对应的是Gneu1

    while (1){

        if(pos - last_pos >= 10000){
            last_pos = pos;
            printf("%cProgress: %.2f%%  ", 13,  (n_thread * (pos - pos_st)) / (float)(n_pair + 1) * 100);
            fflush(stdout);
        }

        struct pair pair = idx2pairs[pos]; //to update
        unsigned short context = pair.prod;
        unsigned short target = pair.token;
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
                target = negative_sampling_table[(next_random >> 16) % table_size];
                /*while(check_pair(target, context) != -1){
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = negative_sampling_table[(next_random >> 16) % table_size];
                }*/
                while(1){
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = negative_sampling_table[(next_random >> 16) % table_size];
                    if(!idx2word[target].prods[context]){
                        break;
                    }
                }
            }

            f = 0;
            for (c = 0; c < layer1_size; c++) f += entity_vector[c + l1] * idx2word[target].vector[c];
            if (f > MAX_EXP) g = (label - 1) * alpha;else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            for (c = 0; c < layer1_size; c++) {
                neu1e[c] += g * idx2word[target].vector[c];
//                if(idx2word[target].vector[c] > 10 || idx2word[target].vector[c] < -10){
//                    printf("neu1e at %d is %f \t current g is %f \t current vector value is %f for target %d, %s \n", c, neu1e[c], g, idx2word[target].vector[c], target, idx2word[target].word);
//                }
            }


        }
        for (c = 0; c < layer1_size; c++) {
            entity_vector[c + l1] += neu1e[c];
//            if(entity_vector[c + l1] > 10 || entity_vector[c + l1] < -10){
//                printf("warning! value \t value:\t %f alpha:\t %f \n", entity_vector[c + l1], alpha);
//            }
        }

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
    char * path = concat(output_file, concat("model_",str));
    printf("Path: %s \n", path);
    FILE *fo;
    fo = fopen(path, "wb");
    long long a, b;
    fprintf(fo, "%lld %lld\n", n_prod + word_size + n_tag, layer1_size);
    for (a = 0; a < n_prod + word_size + n_tag; a++) {
        fprintf(fo, "%s ", idx2prod[a].prod_str);
        if (0) for (b = 0; b < layer1_size; b++)
                fwrite(&entity_vector[a * layer1_size + b], sizeof(float), 1, fo);
        else for (b = 0; b < layer1_size; b++)
                fprintf(fo, "%lf ", entity_vector[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

void train(){
    int i, p;
    pthread_t *pt = (pthread_t *)malloc(n_thread * sizeof(pthread_t));
    for(i = 0; i < 2147400000; i++){
        for (p = 0; p < n_thread; p++) pthread_create(&pt[p], NULL, train_thread, (void *)p);
        for (p = 0; p < n_thread; p++) pthread_join(pt[p], NULL);
        if(i % 10 == 0){
            conclude(i);
        }

//        //save model
//        char str[15];
//        char * path;
//        sprintf(str, "%d", i);
//        path = concat(concat(output_file, str),"_entity_vector");
//        save_model(path, n_prod, layer1_size, entity_vector);
    }
}

int main(int argc, char **argv) {
    init();
    train();

    exit(0);
}