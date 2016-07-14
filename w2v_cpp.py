from w2v_base import W2V_base
from collections import deque

class W2V_cpp(W2V_base):
    def __init__(self, path, folder):
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

    def generate_pos(self):
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = deque(maxlen=span)
        result = []
        f = open("/".join((self.folder, "pair.txt")), "w")

        for obj in self.data:
            text_data = obj["text_data"]
            for text_data_sent in text_data:
                for word_idx in range(0, len(text_data_sent)):
                    word_idx = 0
                    while len(buffer) < span and word_idx < len(text_data_sent):
                        buffer.append(text_data_sent[word_idx])
                        word_idx += 1

                    for word_idx in range(word_idx, len(text_data_sent)):
                        target_idx = int((len(buffer) + 1) / 2)
                        target_word = buffer[target_idx]  # consider buffer is shorter than  self.skip_window

                        for context_word in buffer:
                            if context_word != target_word:
                                result.append((target_word, context_word))

                    #for next word
                    buffer.append(text_data_sent[word_idx])


                if len(result) >= 100000:
                    for target_word, context_word in result:
                        f.write(" ".join((str(target_word), str(context_word))))
                        f.write("\n")
                    result = []

            #write into file
            for target_word, context_word in result:
                f.write(" ".join((str(target_word), str(context_word))))
                f.write("\n")



def main():
    w2c = W2V_cpp("/home/sanqiang/Documents/data/yelp/yelp_academic_dataset_review.json", "yelp_ny_pos")
    w2c.generate_pos()

main()
