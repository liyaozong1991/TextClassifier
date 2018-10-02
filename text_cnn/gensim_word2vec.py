# coding: utf8

import gensim
class SentencesIter(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, encoding='utf8') as f:
            for line in f:
                line_items = line.strip().split('\t')
                if len(line_items) != 2 or len(line_items[1]) == '':
                    continue
                seg_list = line_items[1].split()
                yield seg_list

# train
sentences_iter = SentencesIter('../data/20ng-train-no-stop.txt')
model = gensim.models.Word2Vec(sentences_iter, window=5, min_count=2, workers=10)

model.save('./model/word2vec_gensim')
