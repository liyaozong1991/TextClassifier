# coding: utf8
import sys

label_dict = {}
for line in sys.stdin:
    line = line.strip()
    line_items = line.split()
    label_dict[line_items[0]] = label_dict.get(line_items[0], 0) + 1

for label, num in label_dict.items():
    print(label + '\t' + str(num))
