#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_CODE_LENGTH 40


unsigned int
        layer1_size = 200, *negative_sampling_table, n_thread = 7, n_negative = 10;

char *home = getenv("HOME");
char *path_review = concat(home, "/data/yelp/review_processed.json");

unsigned long long n_prod =   388525 /* 388525 user for prod entity 24974*/, n_pair =  59751978 /*89724193 59751978*/, word_size = 60631;

const int table_size = 1e8;
const int vocab_hash_size = 100000000;