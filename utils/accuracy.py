import numpy
import pdb
import cPickle
import random
import os
import stat
import subprocess
from os.path import isfile, join
from os import chmod
import uuid
import codecs

def conlleval(p, g, w):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    #print os.getcwd()
    filename = os.path.split(os.path.realpath(__file__))[0] + '/tmp/conll_eval_before.txt'

    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = codecs.open(filename, 'w', encoding='utf8')
    f.writelines(out)
    f.close()
    
    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = os.path.split(os.path.realpath(__file__))[0] + '/conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval, ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        #print line
        if 'accuracy' in line:
            out = line.split()
            break
    precision = float(out[3][:-2])
    recall    = float(out[5][:-2])
    f1score   = float(out[7]) / 100

    #if os.path.exists(filename):
    #    os.remove(filename)

    return {'p':precision, 'r':recall, 'f1':f1score}

def get_perfo(filename):
    ''' 
    work around for using a PERL script in python
    dirty but still works.
    '''
    tempfile = str(random.randint(1,numpy.iinfo('i').max)) + '.txt'
    if not isfile(PREFIX + 'conlleval.pl'):
        download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl') 
        chmod('conlleval.pl', stat.S_IRWXU) # give the execute permissions
    if len(PREFIX) > 0:
        chmod(PREFIX + 'conlleval.pl', stat.S_IRWXU) # give the execute permissions
        cmd = PREFIX + 'conlleval.pl < %s | grep accuracy > %s'%(filename,tempfile)
    else:
        cmd = './conlleval.pl < %s | grep accuracy > %s'%(filename,tempfile)
    print cmd
    out = os.system(cmd)
    out = open(tempfile).readlines()[0].split()
    os.system('rm %s'%tempfile)
    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])
    return {'p':precision, 'r':recall, 'f1':f1score}

if __name__ == '__main__':
    print get_perf('valid.txt')
