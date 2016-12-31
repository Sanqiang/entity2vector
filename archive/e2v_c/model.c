//
// Created by sanqiang on 7/31/16.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

_Bool use_model = true;

//useless
void save_model(char * path, unsigned long long n_dataset, unsigned long long layer1_size, float * dataset){
    printf("Save model path: %s \n", path);
    FILE *fo;
    fo = fopen(path, "wb");
    unsigned long long a, b;
    fprintf(fo, "%lld %lld\n", n_dataset, layer1_size);
    for (a = 0; a < n_dataset; a++) {
        //fprintf(fo, "%s ", idx2prod[a].prod_str);
        if (0) for (b = 0; b < layer1_size; b++)
                fwrite(&dataset[a * layer1_size + b], sizeof(float), 1, fo);
        else for (b = 0; b < layer1_size; b++)
                fprintf(fo, "%lf ", dataset[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

float * load_model(char * path){
    printf("Load model path: %s \n", path);
    FILE *fin;
    fin = fopen(path, "rb");
    char ch;
    unsigned int word_idx = 0;
    char word[100];
    unsigned long long integer_num = 0;
    unsigned long long n_datase, layer1_size, a, b;
    float * dataset, decimal_num;

    //for size


    while (!feof(fin)) {
        ch = fgetc(fin);

        if(ch == '\n'){
            layer1_size = integer_num;
            break;
        }else if(ch == ' '){
            n_datase = integer_num;
            integer_num = 0;
        }else{
            integer_num = integer_num * 10;
            integer_num += ch - '0';
        }
    }

    a = posix_memalign((void **)&dataset, 128, (long long)n_datase * layer1_size * sizeof(float));
    if (dataset == NULL) {printf("Memory allocation failed\n"); exit(1);}

    a = 0; b = 0;
    while (!feof(fin)) {
        ch = fgetc(fin);

        if(ch == '\n'){
            word[word_idx] = '\0';
            word_idx = 0;
            decimal_num = atof(word);
            dataset[a*layer1_size+b] = decimal_num;
            a++;
            b = 0;
        }else if(ch == ' '){
            word[word_idx] = '\0';
            decimal_num = atof(word);
            word_idx = 0;
            dataset[a*layer1_size+b] = decimal_num;
            b++;
        }else{
            word[word_idx] = ch;
            word_idx++;
        }
    }

    return dataset;
}

/*
int main(int argc, char **argv) {
    float * dataset;
    unsigned long long a = posix_memalign((void **)&dataset, 128, (long long)3 * 2 * sizeof(float));
    dataset[0] = 0;
    dataset[1] = 1;
    dataset[2] = 2;
    dataset[3] = 3;
    dataset[4] = 4;
    dataset[5] = 5;

    save_model("/home/sanqiang/data/sample/model.txt", 3, 2, dataset);

    float * dataset2 = load_model("/home/sanqiang/data/sample/model.txt");

    printf("%f\n", dataset2[0]);
    printf("%f\n", dataset2[5]);

    exit(0);

    float * dataset = load_model("/Users/zhaosanqiang916/git/entity2vector/yelp_rest_prod_aword/output/model_3");
    printf("%f\n", dataset[0]);
}
*/