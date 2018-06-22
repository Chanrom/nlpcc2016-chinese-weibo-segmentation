#coding=utf-8
import codecs
import sys

filename = sys.argv[1]

lines = codecs.open(filename, 'r', 'utf-8').readlines()
new_lines = []
for i in range(len(lines)):
    line = lines[i].strip()
    if (i == (len(lines) - 1)) and False:
        new_lines.append('')
    else:
        if len(line) == 0:
            new_lines.append('')
        else:
            new_lines.append(line + '\tS')
f = codecs.open('nlpcc2016_test.txt', 'w', 'utf-8')
f.write('\n'.join(new_lines))
f.write('\n\n')
f.close()