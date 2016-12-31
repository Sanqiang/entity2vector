//
// Created by Sanqiang Zhao on 8/10/16.
//
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LENGTH 40

unsigned int
        layer1_size = 200, *negative_sampling_table, n_thread = 1, n_negative = 0;
char vector_file[MAX_STRING] =  "/Users/zhaosanqiang916/git/entity2vector/archive/yelp_rest_prod/wordvector.txt",//"/home/sanqiang/data/glove/glove.twitter.27B.200d.txt"
pair_file[MAX_STRING] =  "/Users/zhaosanqiang916/git/entity2vector/archive/yelp_rest_prod/pairentity.txt", //"/home/sanqiang/git/entity2vector/yelp_rest_allalphaword_yelp_mincnt10_win10/pairentity.txt",
        output_file[MAX_STRING] = "/Users/zhaosanqiang916/git/entity2vector/archive/yelp_rest_prod/output_noneg/", // "/home/sanqiang/git/entity2vector/yelp_rest_allalphaword_yelp_mincnt10_win10/pairentity.txt",
        prod_file[MAX_STRING] = "/Users/zhaosanqiang916/git/entity2vector/archive/yelp_rest_prod/prod.txt"; // "/home/sanqiang/git/entity2vector/yelp_rest_allalphaword_yelp_mincnt10_win10/output.txt";
unsigned long long n_prod =   26629 /* 388525 user for prod entity 24974*/, n_pair =  69242128 /*89724193 59751978*/, word_size = 63258, n_tag= 999;

const int table_size = 1e8;
const int vocab_hash_size = 100000000;