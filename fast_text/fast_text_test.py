# coding: utf8
import tensorflow as tf

word_dict = {}
with open('./model/word_dict') as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        word_dict[line_items[0]] = int(line_items[1])

label_dict = {}
with open('./model/label_dict') as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        label_dict[line_items[0]] = line_items[1]

test_data_path = '../data/20ng-test-no-stop.txt'
words_length = 300
raw_data_record_list = []
with open(test_data_path) as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        if len(line_items) != 2 or line_items[0] == '' or line_items[1] == '':
            continue
        label = line_items[0]
        words = line_items[1].split()
        if len(words) > words_length:
            words = words[:words_length]
        while len(words) < words_length:
            words.append('&')
        raw_data_record_list.append([words, label])

data_record_list = []
for data in raw_data_record_list:
    words = [word_dict.get(k, 0) for k in data[0]]
    labels = label_dict[data[1]]
    data_record_list.append(([words, labels]))

total = 0
correct = 0
with tf.Graph().as_default():
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model/fast_text.model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    for data, label in data_record_list:
        total += 1
        data = [data]
        result = sess.run("predictions:0", feed_dict={"inputs/inputs:0": data})
        if str(result[0]) == str(label):
            correct += 1
        print(str(result[0]) + '\t' + str(label))
    print(correct/total)
