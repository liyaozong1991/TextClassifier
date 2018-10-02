
# coding: utf8

import tensorflow as tf
import collections
import numpy as np
import time

epoches = 10
words_length = 300
n_words = 60000
batch_size = 30
embedding_size = 100
learning_rate = 0.1
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
    data_record_list.append([words, labels])

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
filter_sizes = [3, 4, 5]
num_filters = 4
with graph.as_default():
    # input data
    input_x = tf.placeholder(tf.int64, [None, words_length], name="input_x")
    input_y = tf.placeholder(tf.int64, [None], name="input_y")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
            name="W")
        embedded_chars = tf.nn.embedding_lookup(W, input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, words_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
        W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions")
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = input_y,
                logits = scores
                )
        loss = tf.reduce_mean(losses)
        correct_predictions = tf.equal(predictions, input_y)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acccuray', accuracy)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars)
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        summary_writer = tf.summary.FileWriter('./events', sess.graph)
        index = 0
        for k in range(epoches):
            np.random.shuffle(data_record_list)
            batch_generator = generate_batch(batch_size, data_record_list)
            for x_batch, y_batch in batch_generator:
                index += 1
                feed_dict = {
                    input_x: x_batch,
                    input_y: y_batch,
                    dropout_keep_prob: 0.9
                }
                _, _loss, _accuracy, _summary = sess.run(
                        [train_op, loss, accuracy, summary_op],
                    feed_dict)
                summary_writer.add_summary(_summary, index)

    #saver = tf.train.Saver()
    #saver.save(sess, './model/fast_text.model')
