#coding=utf8
import os
import numpy as np
import random
import cPickle
import codecs
from six.moves import range
import sys
import subprocess
from accuracy import conlleval

def pad_matrix(m, sent_maxlen=None, feature_dim=None, dtype='int32', value=0):
    x = (np.ones((sent_maxlen, feature_dim)) * value).astype(dtype)
    x[:len(m)] = m
    return x



def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0):
    """
        Pad each sequence to the same length:
        the length of the longest sequence.
        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.
        Supports post-padding and pre-padding (default).
        Parameters:
        -----------
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
        Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x
    

def convertNumpy(X):
    '''
    Convert a python list to numpy array.
    '''
    return np.array([np.array(xi).astype('int32') for xi in X])  


def bioEalve(id_sents, id_tagss, id_test_tagss, flag=0, word_dict=None, tag_dict=None):
    ## id_sents, id_tagss, id_labels，前两者都是矩阵，最后一个是向量。三者第一维相等
    assert id_sents.shape == id_tagss.shape
    assert id_sents.shape == id_test_tagss.shape
    #if not os.path.exists(dict_pkl_file):
    #    raise ValueError("Dict file do not exists")
    #word_dict, tag_dict, label_dict = cPickle.load(open(dict_pkl_file, "rb"))

   # evaluation 
    predictions_test = []
    groundtruth_test = []
    words_test = []
    for i in range(len(id_sents)):
        words = []
        groundtruth = []
        predictions = []
        for j in range(len(id_sents[i])):
            if id_sents[i][j] != flag:
                words.append('O' if word_dict[id_sents[i][j]].upper() == '__UNKNOWN__TAG__' else word_dict[id_sents[i][j]].upper())
                groundtruth.append('O' if tag_dict[id_tagss[i][j]].upper() == '__UNKNOWN__TAG__' else tag_dict[id_tagss[i][j]].upper())
                predictions.append('O' if tag_dict[id_test_tagss[i][j]].upper() == '__UNKNOWN__TAG__' else tag_dict[id_test_tagss[i][j]].upper())
        words_test.append(words[:])
        groundtruth_test.append(groundtruth[:])
        predictions_test.append(predictions[:])
    # evaluation // compute the accuracy using conlleval.pl
    res_test = conlleval(predictions_test, groundtruth_test, words_test)
    return res_test

def posEalve(id_sents, gold_masks_seg, masks_seg, id_tagss, id_test_tagss, word_dict=None, tag_dict=None):
    assert id_sents.shape == gold_masks_seg.shape
    assert id_sents.shape == masks_seg.shape
    assert id_sents.shape == id_tagss.shape
    assert id_sents.shape == id_test_tagss.shape

    # unknow word
    #word_dict[1] = u'UNKNOWN'

    #print word_dict
    #print tag_dict

    def gene_pos_sents(id_sents, id_tagss, masks_seg, word_dict, tag_dict):
        # we construct lines like: 我_NN 中国_NN
        sents_and_poss = []
        for i in range(len(id_sents)):
            sent_and_poss = []
            word_and_pos = ''
            for j in range(len(id_sents[i])):
                if id_sents[i][j] == 0: # until padding bit
                    break
                character = word_dict[id_sents[i][j]]
                if masks_seg[i][j] == 0:
                    word_and_pos += character
                if masks_seg[i][j] == 1:
                    word_and_pos += character
                    pos = tag_dict[id_tagss[i][j]]
                    word_and_pos += '_+_' + pos
                    sent_and_poss.append(word_and_pos)
                    word_and_pos = ''
            sents_and_poss.append(' '.join(sent_and_poss))

        return sents_and_poss

    test_sents_poss = gene_pos_sents(id_sents, id_test_tagss, masks_seg, word_dict, tag_dict)
    gs_sents_poss = gene_pos_sents(id_sents, id_tagss, gold_masks_seg, word_dict, tag_dict)

    gs_file = os.path.split(os.path.realpath(__file__))[0] + '/tmp/gs_sents_pos.txt'
    test_file = os.path.split(os.path.realpath(__file__))[0] + '/tmp/test_sents_pos.txt'

    f = codecs.open(gs_file, 'w', 'utf8')
    f.write('\n'.join(gs_sents_poss))
    f.close()

    f = codecs.open(test_file, 'w', 'utf8')
    f.write('\n'.join(test_sents_poss))
    f.close()

    pos_eval = os.path.split(os.path.realpath(__file__))[0] + '/pos_eval.py'

    proc = subprocess.Popen(["python", pos_eval, test_file, gs_file],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    line = proc.stdout.readlines()[-1]
    f1 = (line.strip().split(' '))[-1]

    return float(f1)


def cwsEalve(id_sents, id_tagss, id_test_tagss, word_dict=None, tag_dict=None, test_gs=None, dictionary=None, is_train=True):

        assert id_sents.shape == id_tagss.shape
        assert id_tagss.shape == id_test_tagss.shape
        assert os.path.exists(test_gs)
        assert os.path.exists(dictionary)

        gs_sents = None
        test_sents = None

        # train mode
        if is_train:

            def gene_sent(sents, id_tagss):
                sents_text = []
                for i in range(len(sents)):
                    sent = []
                    word = []
                    sent_length = (1 - np.equal(sents[i], 0)).sum()
                    for j in range(len(sents[i])):
                        if sents[i][j] == 0:
                            break
                        tag = tag_dict[int(id_tagss[i][j])]
                        character = word_dict[int(sents[i][j])]
                        if tag == 'S':
                            #print index
                            sent.append(character)
                        elif tag == 'B':
                            word.append(character)
                            if j == (sent_length - 1): #last tag
                                sent.append(''.join(word))
                                word = []
                        elif tag == 'M':
                            word.append(character)
                            if j == (sent_length - 1): #last tag
                                sent.append(''.join(word))
                                word = []
                        elif tag == 'E':
                            word.append(character)
                            sent.append(''.join(word))
                            word = []
                    sents_text.append('  '.join(sent))
                return sents_text

            gs_sents = gene_sent(id_sents, id_tagss)
            test_sents = gene_sent(id_sents, id_test_tagss)

        else:
            lines = codecs.open(test_gs, 'r', 'utf8').readlines()
            sents = []
            sent = []
            tagss = []
            tags = []
            for line in lines:
                if len(line.strip('\n')) == 0:
                    if len(sent) != 0:
                        sents.append(sent[:])
                        sent = []
                        tagss.append(tags[:])
                        tags = []
                    continue

                l = line.strip('\n').split('\t')
                # print l
                character = l[0]
                gs_tag = l[1]
                sent.append(character)
                tags.append(gs_tag)

            #print id_tagss[0]

            for i in range(len(id_tagss)):
                for j in range(len(tagss[i])):
                    raw_tag = tagss[i][j]
                    id_tag = tag_dict[int(id_tagss[i][j])]
                    # print raw_tag, id_tag,id_tagss[i][j]
                    assert raw_tag == id_tag

            def gene_sent(sents, id_tagss):
                sents_text = []
                for i in range(len(id_tagss)):
                    # print i
                    # print len(sents[i]), (1 - np.equal(id_tagss[i], 0)).sum()
                    # print sents[i]
                    # print id_tagss[i]
                    # assert len(sents[i]) == (1 - np.equal(id_tagss[i], 0)).sum()
                    sent = []
                    word = []
                    for j in range(len(sents[i])):
                        tag = tag_dict[int(id_tagss[i][j])]
                        character = sents[i][j]
                        if tag == 'S':
                            #print index
                            sent.append(character)
                        elif tag == 'B':
                            word.append(character)
                            if j == (len(sents[i]) - 1): #last tag
                                sent.append(''.join(word))
                                word = []
                        elif tag == 'M':
                            word.append(character)
                            if j == (len(sents[i]) - 1): #last tag
                                sent.append(''.join(word))
                                word = []
                        elif tag == 'E':
                            word.append(character)
                            sent.append(''.join(word))
                            word = []
                    sents_text.append('  '.join(sent))
                return sents_text

            gs_sents = gene_sent(sents, id_tagss)
            test_sents = gene_sent(sents, id_test_tagss)

        gs_file = os.path.split(os.path.realpath(__file__))[0] + '/tmp/gs_sents.txt'
        test_file = os.path.split(os.path.realpath(__file__))[0] + '/tmp/test_sents.txt'

        f = codecs.open(gs_file, 'w', 'utf8')
        f.write('\n'.join(gs_sents))
        f.close()

        f = codecs.open(test_file, 'w', 'utf8')
        f.write('\n'.join(test_sents))
        f.close()

        cws_eval = os.path.split(os.path.realpath(__file__))[0] + '/score'

        proc = subprocess.Popen(["perl", cws_eval, dictionary, gs_file, test_file],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        line = proc.stdout.readlines()[-5]
        f1 = (line.strip().split('\t'))[-1]

        return float(f1)

def save_test(id_sents, id_tagss, id_test_tagss, word_dict=None, tag_dict=None, test_gs=None, dictionary=None, is_train=True):

        assert id_sents.shape == id_tagss.shape
        assert id_tagss.shape == id_test_tagss.shape
        assert os.path.exists(test_gs)
        assert os.path.exists(dictionary)

        gs_sents = None
        test_sents = None

        # train mode
        if False:
            pass
        else:
            lines = codecs.open(test_gs, 'r', 'utf8').readlines()
            sents = []
            sent = []
            tagss = []
            tags = []
            for line in lines:
                if len(line.strip('\n')) == 0:
                    if len(sent) != 0:
                        sents.append(sent[:])
                        sent = []
                        tagss.append(tags[:])
                        tags = []
                    continue

                l = line.strip('\n').split('\t')
                #print l
                character = l[0]
                gs_tag = l[1]
                sent.append(character)
                tags.append(gs_tag)

            #print id_tagss[0]

            for i in range(len(id_tagss)):
                for j in range(len(tagss[i])):
                    raw_tag = tagss[i][j]
                    id_tag = tag_dict[int(id_tagss[i][j])]
                    #print raw_tag, id_tag,id_tagss[i][j]
                    assert raw_tag == id_tag

            def gene_sent(sents, id_tagss):
                sents_text = []
                test_sents_text = []
                for i in range(len(id_tagss)):
                    sent = []
                    word = []
                    for j in range(len(sents[i])):
                        tag = tag_dict[int(id_tagss[i][j])]
                        character = sents[i][j]
                        test_sents_text.append(character + '\t' + tag)
                        if tag == 'S':
                            #print index
                            sent.append(character)
                        elif tag == 'B':
                            word.append(character)
                            if j == (len(sents[i]) - 1): #last tag
                                sent.append(''.join(word))
                                word = []
                        elif tag == 'M':
                            word.append(character)
                            if j == (len(sents[i]) - 1): #last tag
                                sent.append(''.join(word))
                                word = []
                        elif tag == 'E':
                            word.append(character)
                            sent.append(''.join(word))
                            word = []
                    sents_text.append('  '.join(sent))
                    test_sents_text.append('')
                return sents_text, test_sents_text

            #gs_sents = gene_sent(sents, id_tagss)
            raw_sents_text, test_sents_text = gene_sent(sents, id_test_tagss)

        # gs_file = os.path.split(os.path.realpath(__file__))[0] + '/tmp/gs_sents.txt'
        test_file = os.path.split(os.path.realpath(__file__))[0] + '/test_my.txt'
        raw_test_file = os.path.split(os.path.realpath(__file__))[0] + '/raw_test_sents.txt'

        f = codecs.open(test_file, 'w', 'utf8')
        f.write('\n'.join(test_sents_text))
        f.close()

        f = codecs.open(raw_test_file, 'w', 'utf8')
        f.write('\n'.join(raw_sents_text))
        f.close()

        # cws_eval = os.path.split(os.path.realpath(__file__))[0] + '/score'

        # proc = subprocess.Popen(["perl", cws_eval, dictionary, gs_file, test_file],
        #     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        # line = proc.stdout.readlines()[-5]
        # f1 = (line.strip().split('\t'))[-1]

        # return float(f1)



def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:                                 #半角空格直接转化                  
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:        #半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += unichr(inside_code)
    return rstring    

def id2original(id_sents, id_tagss, id_labels, id_test_tagss, id_test_labels, flag=0, word_dict=None, tag_dict=None, label_dict=None, output_file=''):

    ## id_sents, id_tagss, id_labels，前两者都是矩阵，最后一个是向量。三者第一维相等
    assert id_sents.shape == id_tagss.shape
    assert len(id_sents) == len(id_labels)
    assert id_sents.shape == id_test_tagss.shape
    assert len(id_sents) == len(id_test_labels)
    #if not os.path.exists(dict_pkl_file):
    #    raise ValueError("Dict file do not exists")
    #word_dict, tag_dict, label_dict = cPickle.load(open(dict_pkl_file, "rb"))

    if len(id_labels.shape) == 1:
        # every item represent a sentence instance, include
        # the sentence, gs chunk label, test chunk label, gs/test sentiment label
        # every item represented by a dict, its keys: 'words', 'gs_c_labels', 'test_c_labels', 'gs_test_st_labels'
        items = []
        for i in range(len(id_sents)):
            item = {}
            words = []
            gs_c_labels = []
            test_c_labels = []
            gs_test_st_labels = []
            try:
                for j in range(len(id_sents[i])):
                    if int(id_sents[i][j]) != int(flag):
                        assert int(id_tagss[i][j]) != int(flag)
                        words.append(word_dict[int(id_sents[i][j])])
                        gs_c_labels.append(tag_dict[int(id_tagss[i][j])])
                        test_c_labels.append(tag_dict[int(id_test_tagss[i][j])])
                gs_test_st_labels.append(label_dict[int(id_labels[i])] + ' / ' + label_dict[int(id_test_labels[i])])

                item['words'] = words[:]
                item['gs_c_labels'] = gs_c_labels[:]
                item['test_c_labels'] = test_c_labels[:]
                item['gs_test_st_labels'] = gs_test_st_labels[:]
            
                items.append(item)
            except:
                print "\nid2original Unexpected error:", sys.exc_info()[0]

        s = []
        #print items[0]
        for item in items:
            words = ' '.join((i if (len(i) < 12) else i[:12]).ljust(12) for i in item['words'])
            gs_c_labels = ' '.join((i if (len(i) < 12) else i[:12]).ljust(12) for i in item['gs_c_labels'])
            test_c_labels = ' '.join((i if (len(i) < 12) else i[:12]).ljust(12) for i in item['test_c_labels'])
            gs_test_st_labels = ' '.join(item['gs_test_st_labels'])
            s.append(words + '\n' + gs_c_labels + '\n' + test_c_labels + '\n' + gs_test_st_labels + '\n')

        f = codecs.open(output_file, 'w', encoding='utf-8')
        f.write('\n'.join(s))
        f.close()

    if len(id_labels.shape) == 2:
        # every item represent a sentence instance, include
        # the sentence, gs chunk label, test chunk label, gs/test sentiment label
        # every item represented by a dict, its keys: 'words', 'gs_c_labels', 'test_c_labels', 'gs_test_st_labels'
        items = []
        for i in range(len(id_sents)):
            item = {}
            words = []
            gs_c_labels = []
            test_c_labels = []
            gs_st_labels= []
            test_st_labels = []
            try:
                for j in range(len(id_sents[i])):
                    if int(id_sents[i][j]) != int(flag):
                        assert int(id_tagss[i][j]) != int(flag)
                        words.append(word_dict[int(id_sents[i][j])])
                        gs_c_labels.append(tag_dict[int(id_tagss[i][j])])
                        test_c_labels.append(tag_dict[int(id_test_tagss[i][j])])
                        gs_st_labels.append(label_dict[int(id_labels[i][j])])
                        test_st_labels.append(label_dict[int(id_test_labels[i][j])])
                
                item['words'] = words[:]
                item['gs_c_labels'] = gs_c_labels[:]
                item['test_c_labels'] = test_c_labels[:]
                item['gs_st_labels'] = gs_st_labels[:]
                item['test_st_labels'] = test_st_labels[:]
            
                items.append(item)
            except:
                print "\nid2original Unexpected error:", sys.exc_info()[0]

        s = []
    #    print items[0]
        for item in items:
            words = ' '.join((i if (len(i) < 5) else i[:5]).ljust(5) for i in item['words'])
            gs_c_labels = ' '.join((i if (len(i) < 5) else i[:5]).ljust(5) for i in item['gs_c_labels'])
            test_c_labels = ' '.join((i if (len(i) < 5) else i[:5]).ljust(5) for i in item['test_c_labels'])
            gs_st_labels = ' '.join((i if (len(i) < 5) else i[:5]).ljust(5) for i in item['gs_st_labels'])
            test_st_labels = ' '.join((i if (len(i) < 5) else i[:5]).ljust(5) for i in item['test_st_labels'])
            s.append(strB2Q(words + '\n' + gs_c_labels + '\n' + test_c_labels + '\n' + gs_st_labels + '\n' + test_st_labels + '\n'))

        f = codecs.open(output_file, 'w', encoding='utf-8')
        f.write('\n'.join(s))
        f.close()


if __name__ == '__main__':

    #train_inputs, train_gs_chunk_label, train_chunk_label, word_dict, tag_dict, GS_FILE, DICTIONARY, is_train = cPickle.load(open('s', 'rb'))

    #cwsEalve(train_inputs, train_gs_chunk_label, train_chunk_label, word_dict, tag_dict, GS_FILE, DICTIONARY, is_train)
    train_inputs, train_masks_seg, train_gs_sent_label, train_sent_label, word_dict, label_dict = cPickle.load(open('s', 'rb'))

    #print word_dict
    #print label_dict

    print posEalve(train_inputs, train_masks_seg, train_gs_sent_label, train_sent_label,
             word_dict, label_dict)

