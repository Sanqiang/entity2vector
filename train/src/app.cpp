#include "data.h"
#include <fstream>
#include "util.h"
#include "controller.h"

using namespace entity2vec;


int main(int argc, char** argv) {
    util::initTables();
    std::shared_ptr<args> a = std::make_shared<args>();

    controller con;
    con.train(a);

    exit(0);

}