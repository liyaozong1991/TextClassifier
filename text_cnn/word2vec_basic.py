# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# refer: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
"""Basic word2vec example."""
import collections
import math
import tensorflow as tf
import numpy as np

max_train_steps = 1000000
batch_size = 50
skip_window = 2
vocabulary_size = 60000
embedding_size = 100  # Dimension of the embedding vector.
num_sampled = 64  # Number of negative examples to sample.
learning_rate = 0.1
gpu_nums = 8
train_data_path = '../data/20ng-train-no-stop.txt'
data_list = []
raw_words_list = []
with open(train_data_path) as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        if len(line_items) != 2 or line_items[0] == '' or line_items[1] == '':
            continue
        words = line_items[1].split()
        data_list.append(words)
        raw_words_list.extend(words)

# Step 2: Build the dictionary and replace rare words with UNK token.
count = collections.Counter(raw_words_list).most_common(vocabulary_size-1)
word_dict = {'UNK': 0}
for word, _ in count:
    word_dict[word] = len(word_dict)

words_id_list = []
for words in data_list:
    word_id_list = []
    for word in words:
        word_id_list.append(word_dict.get(word, 0))
    words_id_list.append(word_id_list)

def generate_train_batch(batch_size, words_id_list, skip_window):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    index = 0
    for word in words_id_list:
        for i in range(len(word)):
            context_words = [k for k in range(max(0, i-skip_window), min(len(word), i+skip_window+1)) if k != i]
            words_to_use = context_words
            for context_word in words_to_use:
                batch[index] = word[i]
                labels[index][0] = word[context_word]
                index += 1
                if index == batch_size:
                    yield batch, labels
                    index = 0

def get_next_batch(words_id_list):
    while True:
        np.random.shuffle(words_id_list)
        generate_batch = generate_train_batch(batch_size, words_id_list, skip_window)
        for batch, labels in generate_batch:
            yield batch, labels

generate_batch = get_next_batch(words_id_list)

# Step 4: Build and train a skip-gram model.
def inference(batch):
    with tf.device('/cpu:0'):
        # Input data.
        # train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_inputs = tf.convert_to_tensor(batch)
        shape=[vocabulary_size, embedding_size]
        # Look up embeddings for inputs.
        embeddings = tf.get_variable(
            name='embeddings',
            dtype=tf.float32,
            initializer=tf.random_uniform(
                shape=shape,
                minval=-1.0,
                maxval=1.0,
                )
            )
        embed = tf.nn.embedding_lookup(
                embeddings,
                train_inputs,
                )
        nce_weights = tf.get_variable(
            name='nce_weights',
            initializer=tf.truncated_normal(
                shape,
                stddev=1.0 / math.sqrt(embedding_size),
                ),
            )
        nce_biases = tf.get_variable(
                name='nce_biases',
                initializer=tf.zeros([vocabulary_size]),
                )
        return embed, nce_weights, nce_biases

def cal_loss(logits, train_labels, nce_weights, nce_biases):
    train_labels = tf.convert_to_tensor(train_labels, dtype=np.int64)
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=logits,
            num_sampled=num_sampled,
            num_classes=vocabulary_size,
        ),
    )
    return loss

def tower_loss(scope):
    batch, labels = next(generate_batch)
    logits, nce_weights, nce_biases = inference(batch)
    local_loss = cal_loss(logits, labels, nce_weights, nce_biases)
    return local_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    tower_grads = []
    for i in range(gpu_nums):
        with tf.device('/gpu:{}'.format(i)):
            with tf.name_scope('word2vec.tower_{}'.format(i)) as scope:
                loss = tower_loss(scope)
                # important
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(loss)
                tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    # Apply the gradients to adjust the shared variables.
    train_op = opt.apply_gradients(grads)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)
    sess.run(init)
    for i in range(max_train_steps):
        sess.run([train_op])
        print(loss_value)
    #embeddings = embeddings.eval(sess)
    #with open('./model/word2vec_multi', 'w') as fw:
    #    for word in word_dict:
    #        fw.write('\t'.join([word, str(word_dict[word]), ','.join(str(k) for k in embeddings[word_dict[word]])]) + '\n')
train()
