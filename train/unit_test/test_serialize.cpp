#include "../src/data.h"
#include <fstream>
#include "../src/util.h"
#include "../src/controller.h"
#include "assert.h"

using namespace entity2vec;

int main(int argc, char** argv){
    util::initTables();
    std::shared_ptr<args> a = std::make_shared<args>();

    a->load_model_flag = 0;
    std::shared_ptr<controller> con1 = std::make_shared<controller>();
    con1->train(a);
    con1->saveModel("unit");

    std::shared_ptr<controller> con2 = std::make_shared<controller>();
    con2->args_ = a;
    con2->loadModel("unit");

    //matrix input
    assert(con1->input_->m_ == con2->input_->m_);
    assert(con1->input_->n_ == con2->input_->n_);
    for (int64_t i = 0; i < con1->input_->m_; ++i) {
        for (int64_t j = 0; j < con1->input_->n_; ++j) {
            assert(con1->input_->data_[i*con1->input_->n_ + j] == con2->input_->data_[i*con1->input_->n_ + j]);
        }
    }

    //matrix input
    assert(con1->output_->m_ == con2->output_->m_);
    assert(con1->output_->n_ == con2->output_->n_);
    for (int64_t i = 0; i < con1->output_->m_; ++i) {
        for (int64_t j = 0; j < con1->output_->n_; ++j) {
            assert(con1->output_->data_[i*con1->input_->n_ + j] == con2->output_->data_[i*con1->input_->n_ + j]);
        }
    }

    //data
    assert(con1->data_->word_size_ == con2->data_->word_size_);
    assert(con1->data_->prod_size_ == con2->data_->prod_size_);
    for (int64_t i = 0; i < con1->data_->word_size_; ++i) {
        assert(con1->data_->idx2words_[i].word == con2->data_->idx2words_[i].word);
        assert(con1->data_->idx2words_[i].prod_id == con2->data_->idx2words_[i].prod_id);
        assert(con1->data_->idx2words_[i].count == con2->data_->idx2words_[i].count);
    }

    for (int64_t i = 0; i < con1->data_->VOCAB_HASH_SIZE; ++i) {
        assert(con1->data_->word2idx_[i] == con2->data_->word2idx_[i]);
    }

    for (int64_t i = 0; i < con1->data_->prod_size_; ++i) {
        assert(con1->data_->idx2prod_[i].prod == con2->data_->idx2prod_[i].prod);
        assert(con1->data_->idx2prod_[i].count == con2->data_->idx2prod_[i].count);
        assert(con1->data_->idx2prod_[i].word_count == con2->data_->idx2prod_[i].word_count);

        for (int64_t j = 0; j < con1->data_->idx2prod_[i].word_count; ++j) {
            assert(con1->data_->idx2prod_[i].idx2words_[j] == con2->data_->idx2prod_[i].idx2words_[j]);
        }

        for (int64_t j = 0; j < con1->data_->SUB_VOCAB_HASH_SIZE; ++j) {
            assert(con1->data_->idx2prod_[i].word2idx_[j] == con2->data_->idx2prod_[i].word2idx_[j]);
        }
    }

    for (int64_t i = 0; i < con1->data_->PROD_HASH_SIZE; ++i) {
        assert(con1->data_->prod2idx_[i] == con2->data_->prod2idx_[i]);
    }


}

