# coding:utf8
import numpy as np

word_vec_dict = {}
with open('./model/word2vec') as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        word_vec_dict[line_items[0]] = np.fromstring(line_items[2], sep=',')

a = word_vec_dict['story']
distance_list = []
for word in word_vec_dict:
    distance_list.append([word, np.linalg.norm(a-word_vec_dict[word])])

distance_list = list(sorted(distance_list, key=lambda k:k[1]))
for item in distance_list[:20]:
    print(item[0], item[1])
