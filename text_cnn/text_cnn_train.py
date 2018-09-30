# coding: utf8
import tensorflow as tf
import time
import numpy as np

words_length = 300
embedding_length = 100
epoches = 10
batch_size = 30
learning_rate = 0.01

# load word vec
word_vec_dict = {}
with open('./model/word2vec') as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        word_vec_dict[line_items[0]] = [float(k) for k in line_items[2].split(',')]

# read train data
train_data_path = '../data/20ng-train-no-stop.txt'
words_list = []
label_set = set()
with open(train_data_path) as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
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
        vec = word_vec_dict.get(word, word_vec_dict['UNK'])
        local_vec.append(vec)
    if len(local_vec) > words_length:
        local_vec = local_vec[:words_length]
    while len(local_vec) < words_length:
        local_vec.append(word_vec_dict['UNK'])
    word_vec_list.append([label_id_dict[label], local_vec])


def generate_batch(words_vec_list, batch_num):
    batch, label = [], []
    for word in words_vec_list:
        label.append(word[0])
        batch.append(word[1])
        if len(label) == batch_num:
            yield batch, label
            batch, label = [], []


with tf.Graph().as_default():
    inputs = tf.placeholder(tf.float64, shape=[None, words_length, embedding_length])
    labels = tf.placeholder(tf.int32, shape=[None])
    inputs_r = tf.reshape(inputs, [-1, words_length, embedding_length, 1])

    def get_pool(filters, size):
        conv = tf.layers.conv2d(
            inputs=inputs_r,
            filters=filters,
            kernel_size=[size, embedding_length],
            padding='valid',
            activation=tf.nn.relu,
            use_bias=True,
        )
        pool = tf.layers.max_pooling2d(
            inputs=conv,
            pool_size=[words_length-size+1, 1],
            strides=1,
        )
        return pool

    pool2 = get_pool(6, 2)
    pool3 = get_pool(7, 3)
    pool4 = get_pool(8, 4)
    pool5 = get_pool(9, 5)
    pool = tf.concat(
        values=[pool2, pool3, pool4, pool5],
        axis=3
    )
    pool = tf.reshape(pool, [-1, 30])
    #logits = tf.contrib.layers.fully_connected(
    #    inputs=pool,
    #    num_outputs=num_classes,
    #    activation_fn=None,
    #)
    logits = tf.layers.dense(
        inputs=pool,
        units=num_classes,
    )
    predictions = tf.argmax(logits, axis=-1, name='predictions')
    mean_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits,
        ),
    )
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_loss, global_step=tf.train.get_global_step())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    num = 0
    start_time = time.time()
    for k in range(epoches):
        np.random.shuffle(word_vec_list)
        batch_generator = generate_batch(word_vec_list, batch_size)
        for batch, label in batch_generator:
            num += 1
            loss, _ = sess.run([mean_loss, train_step], feed_dict={inputs: batch, labels: label})
            print('The {}th train is over, loss: {}'.format(num, loss))
    end_time = time.time()
    print('total time:{}'.format(end_time - start_time))

    saver = tf.train.Saver()
    saver.save(sess, './model/text_cnn.model')
