# coding: utf8
import tensorflow as tf
import numpy as np
import gensim

words_length = 300
embedding_length = 100
epoches = 10
batch_size = 30
learning_rate = 0.01

# load word vec
#word_vec_dict = {}
#with open('./model/word2vec') as f:
#    for line in f:
#        line = line.strip()
#        line_items = line.split('\t')
#        word_vec_dict[line_items[0]] = [float(k) for k in line_items[2].split(',')]
word_vec_dict = gensim.models.Word2Vec.load("./model/word2vec_gensim")

# read train data
test_data_path = '../data/20ng-test-no-stop.txt'
words_list = []
label_set = set()
with open(test_data_path) as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        if len(line_items) != 2 or line_items[0] == '' or line_items[1] == '':
            continue
        label = line_items[0]
        label_set.add(label)
        words = line_items[1].split()
        words_list.append([label, words])

num_classes = len(label_set)
label_id_dict = {}
for label in label_set:
    label_id_dict[label] = len(label_id_dict)

word_vec_list = []
for word_list in words_list:
    local_vec = []
    label = word_list[0]
    for word in word_list[1]:
        #vec = word_vec_dict.get(word, word_vec_dict['UNK'])
        if word in word_vec_dict.wv:
            vec = word_vec_dict.wv[word]
        else:
            vec = np.zeros(100)
        local_vec.append(vec)
    if len(local_vec) > words_length:
        local_vec = local_vec[:words_length]
    while len(local_vec) < words_length:
        # local_vec.append(word_vec_dict['UNK'])
        local_vec.append(np.zeros(100))
    word_vec_list.append([label_id_dict[label], local_vec])

total = 0
correct = 0
with tf.Graph().as_default():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model/text_cnn.model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    for label, vec in word_vec_list:
        total += 1
        vec = [vec]
        result = sess.run("predictions:0", feed_dict={"inputs:0": vec})
        if str(result[0]) == str(label):
            correct += 1
        print(str(result[0]) + '\t' + str(label))
    print(correct/total)

