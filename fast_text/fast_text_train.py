# coding: utf8

import tensorflow as tf
import collections
import numpy as np

epoches = 10
words_length = 300
n_words = 60000
batch_size = 30
embedding_size = 10
learning_rate = 0.01
train_data_path = '../data/20ng-train-no-stop.txt'

raw_data_record_list = []
words_list = []
label_set = set()
with open(train_data_path) as f:
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
        words_list.extend(words)
        label_set.add(label)
        raw_data_record_list.append([words, label])

# build dictionary
count = collections.Counter(words_list).most_common(n_words)
word_dict = {}
# every word with unique id
for word, _ in count:
    word_dict[word] = len(word_dict)

with open('./model/word_dict', 'w') as fw:
    for word, ids in word_dict.items():
        fw.write(word + '\t' + str(ids) + '\n')

label_dict = {}
for label in label_set:
    label_dict[label] = len(label_dict)

with open('./model/label_dict', 'w') as fw:
    for word, ids in label_dict.items():
        fw.write(word + '\t' + str(ids) + '\n')

data_record_list = []
for data in raw_data_record_list:
    words = [word_dict.get(k, 0) for k in data[0]]
    labels = label_dict[data[1]]
    data_record_list.append(([words, labels]))

# generate train data
def generate_batch(batch_size, data_record_list):
    for i in range(0, len(data_record_list), batch_size):
        batch = [k[0] for k in data_record_list[i:i+batch_size]]
        labels = [k[1] for k in data_record_list[i:i+batch_size]]
        yield batch, labels

# build graph
vocabulary_size = len(word_dict)
num_classes = len(label_dict)
graph = tf.Graph()
with graph.as_default():
    # input data
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[None, words_length], name='inputs')
        train_labels = tf.placeholder(tf.int32, shape=[None], name='labels')

    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.reduce_mean(tf.nn.embedding_lookup(embeddings, train_inputs), 1)
        input_layer = embed
        logits = tf.contrib.layers.fully_connected(
            inputs=input_layer,
            num_outputs=num_classes,
            activation_fn=None,
        )
        predictions = tf.argmax(logits, axis=-1, name='predictions')
        mean_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=train_labels,
                logits=logits,
            ),
        )
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_loss, global_step=tf.train.get_global_step())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    num = 0
    for k in range(epoches):
        np.random.shuffle(data_record_list)
        batch_generator = generate_batch(batch_size, data_record_list)
        for batch, labels in batch_generator:
            num += 1
            loss, _ = sess.run([mean_loss, train_step], feed_dict={train_inputs: batch, train_labels: labels})
            print('The {}th train is over, loss: {}'.format(num, loss))

    saver = tf.train.Saver()
    saver.save(sess, './model/fast_text.model')