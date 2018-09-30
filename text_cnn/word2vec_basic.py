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

epoches = 10
batch_size = 50
skip_window = 2
vocabulary_size = 60000
embedding_size = 100  # Dimension of the embedding vector.
num_sampled = 64  # Number of negative examples to sample.
learning_rate = 0.1 # Step 1: read data.
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
    batch,labels = [], []
    for word in words_id_list:
        for i in range(len(word)):
            context_words = [k for k in range(max(0, i-skip_window), min(len(word), i+skip_window+1)) if k != i]
            words_to_use = context_words
            for context_word in words_to_use:
                batch.append(word[i])
                labels.append([word[context_word]])
                if len(batch) == batch_size:
                    yield batch, labels
                    batch, labels = [], []


# Step 4: Build and train a skip-gram model.
graph = tf.Graph()

with graph.as_default():
    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(
                tf.random_uniform(
                    [vocabulary_size, embedding_size],
                    -1.0,
                    1.0,
                ),
                name='embeddings'
            )
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    with tf.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, embedding_size],
                stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size,
            ),
        )
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # Add variable initializer.
    init = tf.global_variables_initializer()
    sess = tf.Session()
    # We must initialize all variables before we use them.
    sess.run(init)
    for k in range(epoches):
        np.random.shuffle(words_id_list)
        batch_generator = generate_train_batch(batch_size, words_id_list, skip_window)
        for batch, labels in batch_generator:
            feed_dict = {train_inputs: batch, train_labels: labels}
            _, loss_val = sess.run(
                [optimizer, loss],
                feed_dict=feed_dict,
            )
            print(loss_val)
    # Save the model for checkpoints.
    embeddings = embeddings.eval(sess)
    with open('./model/word2vec', 'w') as fw:
        for word in word_dict:
            fw.write('\t'.join([word, str(word_dict[word]), ','.join(str(k) for k in embeddings[word_dict[word]])]) + '\n')
