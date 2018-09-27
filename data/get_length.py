# coding: utf8
import sys

total_length = 0
total_num = 0
for line in sys.stdin:
    line = line.strip()
    line_items = line.split()
    total_length += len(line_items)
    total_num += 1

print(total_length / total_num)
