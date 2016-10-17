//
// Created by Sanqiang Zhao on 10/10/16.
//

#include "util.h"
#include "real.h"
#include <cmath>
#include <fstream>

namespace entity2vec {
namespace util{
    real *t_sigmoid = nullptr;
    real *t_log = nullptr;

    real log(real x) {
        if (x > 1.0) {
            return 0.0;
        }
        int i = int(x * LOG_TABLE_SIZE);
        return t_log[i];
    }

    real sigmoid(real x) {
        if (x < -MAX_SIGMOID) {
            return 0.0;
        } else if (x > MAX_SIGMOID) {
            return 1.0;
        } else {
            int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
            return t_sigmoid[i];
        }
    }

    void initSigmoid() {
        if (t_sigmoid != nullptr) return;
        t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
        for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
            real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
            t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
        }
    }

    void initLog() {
        if (t_log != nullptr) return;
        t_log = new real[LOG_TABLE_SIZE + 1];
        for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
            real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
            t_log[i] = std::log(x);
        }
    }

    void initTables() {
        initLog();
        initSigmoid();
    }

    void seek(std::ifstream &ifs, uint64_t pos) {
        ifs.clear();
        ifs.seekg(std::streampos(pos));
    }
}
}