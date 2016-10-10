#include "data.h"
#include <fstream>

using namespace entity2vec;

int main(int argc, char** argv) {
    std::string path = "/Users/zhaosanqiang916/data/yelp/review_processed_sample.txt";
    std::ifstream ifs(path);
    data d;
    d.readFromFile(ifs);
    ifs.close();
}