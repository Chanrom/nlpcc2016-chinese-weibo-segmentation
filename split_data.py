#coding=utf-8
import codecs
import sys
import numpy as np

## split data to train and development set (9:1)
## usage: python split_data.py <train file>

filename = sys.argv[1]
name = 'nlpcc2016'

split = 0.1

lines = codecs.open(filename, 'r', 'utf-8').readlines()
index_range = []
prev_i = 0
print 'total num line', len(lines)
for i in range(len(lines)):
    line = lines[i].strip()
    #if (len(line) == 0 and i != (len(lines) - 1)):
    if (len(line) == 0):
        index_range.append((prev_i, i - 1))
        prev_i = i + 1

np.random.seed(1)
np.random.shuffle(index_range)
size = len(index_range)
print 'total samples num', size
x1 = index_range[:int(size * (1 - split))]
print 'splited train set size', len(x1)
x2 = index_range[int(size * (1 - split)):]
print 'splited dev set size', len(x2)

f = codecs.open(name + '_train.txt', 'w', 'utf-8')
s = []
for item in x1:
    for i in range(item[0], item[1] + 1):
        s.append(lines[i])
    s.append('\n')
f.write(''.join(s))
f.close()

f = codecs.open(name + '_dev.txt', 'w', 'utf-8')
s = []
for item in x2:
    for i in range(item[0], item[1] + 1):
        s.append(lines[i])
    s.append('\n')
f.write(''.join(s))
f.close()
