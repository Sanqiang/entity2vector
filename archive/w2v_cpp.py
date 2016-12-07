#word2vec pair generator
from w2v_base import W2V_base
from collections import deque
from struct import Struct
from collections import Counter


class W2V_cpp(W2V_base):
    def __init__(self, path, folder):
        print(folder)
        W2V_base.__init__(self, path, folder)

    '''
    def init_unigram_table(self):
        table_size = 1e8
        self.table = [0] * table_size

        power = 0.75
        words_pow = 0
        for word, cnt in self.word_count:
            words_pow += cnt^power
        i = 0
        words = list(self.word_count.keys())
        d1 = self.word_count[words[i]]^power / words_pow
        for a in range(table_size):
            self.table[i] = a
            if a / table_size > d1:
                i+=1
                d1 = self.word_count[words[i]]^power / words_pow
            if i >= self.vocab_size:
                i = self.vocab_size - 1

        return self.table
    '''

    def countlines(self):
        cnt = 0
        f = open("/".join((self.folder, "pair.txt")), "r")
        for line in f:
            cnt += 1
        return cnt

    def generate_word(self):
        f = open("/".join((self.folder, "pairword.txt")), "w")
        for word, cnt in self.word_count:
            #cnt = self.word_count[word]
            f.write(word)
            f.write(" ")
            f.write(str(cnt))
            f.write("\n")

    def generate_pos(self):
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]

        result = []
        f = open("/".join((self.folder, "pair.txt")), "w")

        for obj in self.data:
            text_data = obj["text_data"]
            for text_data_sent in text_data:
                text_data_sent = [[items[1] for items in sent] for sent in text_data_sent]
                # process each sentence
                word_idx = 0
                buffer = deque(maxlen=span)

                while len(buffer) < span and word_idx < len(text_data_sent):
                    buffer.append(text_data_sent[word_idx])
                    word_idx += 1

                for target_idx in range(0, int((len(buffer)) / 2)):
                    target_word = buffer[target_idx]
                    if target_word != -1:
                        for context_word in buffer:
                            if context_word != target_word and context_word != -1:
                                result.append((target_word, context_word))

                for word_idx in range(word_idx, len(text_data_sent)):
                    target_idx = int((len(buffer)) / 2)
                    target_word = buffer[target_idx]  # consider buffer is shorter than  self.skip_window

                    if target_word != -1:
                        for context_word in buffer:
                            if context_word != target_word and context_word != -1:
                                result.append((target_word, context_word))

                    # for next word
                    buffer.append(text_data_sent[word_idx])
                    word_idx += 1

                for target_idx in range(int((len(buffer)) / 2), len(buffer)):
                    target_word = buffer[target_idx]
                    if target_word != -1:
                        for context_word in buffer:
                            if context_word != target_word and context_word != -1:
                                result.append((target_word, context_word))

            if len(result) >= 1000000:
                for target_word, context_word in result:
                    f.write(" ".join((str(target_word), str(context_word))))
                    f.write("\n")
                print(len(result))
                result = []

        # write into file
        for target_word, context_word in result:
            f.write(" ".join((str(target_word), str(context_word))))
            f.write("\n")
        print(len(result))


def main():
    #w2c = W2V_cpp("/home/sanqiang/data/yelp/review.json", "yelp_allalphaword_mincnt10_win10")
    #w2c = W2V_cpp("/Users/zhaosanqiang916//data/yelp/review.json", "yelp_ny_pos")
    w2c = W2V_cpp("/home/sanqiang/data/yelp/review.json", "yelp_allalphaword_mincnt5_win10")
    w2c.generate_word()
    w2c.generate_pos()
    print(w2c.countlines())


main()
