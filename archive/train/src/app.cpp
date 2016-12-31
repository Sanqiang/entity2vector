#include "data.h"
#include <fstream>
#include "util.h"
#include "controller.h"

using namespace entity2vec;



void train(){
    util::initTables();
    std::shared_ptr<args> a = std::make_shared<args>();

    controller con;
    con.train(a);
}

void generate_vectors(){
    util::initTables();
    std::shared_ptr<args> a = std::make_shared<args>();

    controller con;
    con.args_ = a;
    con.loadModel("test_addpre2880001");
    con.saveVectors("test0");
}

int main(int argc, char** argv) {
    train();

    exit(0);

}