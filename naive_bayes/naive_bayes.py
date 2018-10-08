# coding: utf8
import math
total_train_num = 0
label_num_dict = {}
word_label_num_dict = {}
label_word_count_dict = {}
word_set = set()

train_data = '../data/20ng-train-no-stop.txt'
with open(train_data) as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        if len(line_items) != 2 or line_items[0] == '' or line_items[1] == '':
            continue
        total_train_num += 1
        label = line_items[0]
        words = line_items[1].split()
        label_num_dict[label] = label_num_dict.get(label, 0) + 1
        for word in words:
            word_set.add(word)
            word_label_num_dict[word+'_'+label] = word_label_num_dict.get(word+'_'+label, 0) + 1
            label_word_count_dict[label] = label_word_count_dict.get(label, 0) + 1

# for laplace smoothing
for word in word_set:
    for label in label_num_dict:
        word_label_num_dict[word+'_'+label] = word_label_num_dict.get(word+'_'+label, 0) + 1
        label_word_count_dict[label] = label_word_count_dict.get(label, 0) + 1

def get_label_prob(label, words_list):
    # priority prob
    p = math.log(label_num_dict.get(label) / total_train_num)
    for word in words_list:
        if word not in word_set:
            continue
        p += math.log((word_label_num_dict.get(word+'_'+label) / label_word_count_dict[label]))
    return p

test_data = '../data/20ng-test-no-stop.txt'
test_total = 0
correct_num = 0
with open(test_data) as f:
    for line in f:
        line = line.strip()
        line_items = line.split('\t')
        if len(line_items) != 2 or line_items[0] == '' or line_items[1] == '':
            continue
        test_total += 1
        real_label = line_items[0]
        words = line_items[1].split()
        predict_list = []
        for label in label_num_dict:
            predict_list.append([label, get_label_prob(label, words)])
        predict_label = max(predict_list, key=lambda x:x[1])[0]
        if predict_label == real_label:
            correct_num += 1

print(correct_num / test_total)
