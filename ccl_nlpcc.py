#coding=utf8
#from base.base import ReadData
import os
import sys
import numpy as np
import cPickle
import codecs
import time
from collections import OrderedDict

class NLPCC(object):
    '''
        Process NLPCC 2015 Seg&POS shared task data. The train data is like this:
        "word1<tab>Seg<tab>POS
         word2<tab>Seg<tab>POS

         word_new_sent1<tab>POS<tab>tag1"
        that means every line start with a word, then follows a tab,
        then there is POS tag , and then a chunking tag.
        when comes to a new sentence, empty line is created.
    '''
    def __init__(self, word_dict=None, tag_dict=None, label_dict=None):

        self.max_length = 0
        self.max_labels = 0
        if isinstance(word_dict, OrderedDict):
            self.word_dict = word_dict
        else:
            self.word_dict = OrderedDict()
            self.word_dict['_unknown_word_'] = 1
        if isinstance(label_dict, OrderedDict):
            self.label_dict = label_dict
        else:
            self.label_dict = OrderedDict()
            #self.label_dict['__unknown__tag__'] = 1
        if isinstance(tag_dict, OrderedDict):
            self.tag_dict = tag_dict
        else:
            self.tag_dict = OrderedDict()
            self.tag_dict['B'] = 1
            self.tag_dict['E'] = 2
            self.tag_dict['S'] = 3
            self.tag_dict['M'] = 4
            #self.tag_dict['__unknown__tag__'] = 1

        self.sents = None
        self.tagss = None
        self.labels = None

        self.av_features = None

        self.lex_features = None

        self.en_features = None

        self.seen_words_num = 0
        self.seen_tags_num = 0
        self.seen_labels_num = 0

        self.unseen_words = None
        self.unseen_tags = None
        self.unseen_labels = None

        self.tag_number = 0
        self.label_number = 0

    def read_file(self, filename):
        if not os.path.exists(filename):
            raise IOError("Input file do not exists")
        lines = codecs.open(filename, 'r', encoding='utf8').readlines()
        return [line.strip('\n') for line in lines]

    def convertNumpy(self, X):
        '''
            Convert a python list to numpy array.
        '''
        return np.array([np.array(xi).astype('int32') for xi in X])   

    def process(self, filename, train=True, split=' '):
        print 'Processing...'
        self.word2idx(self.read_file(filename), train, split)     

    def word2idx(self, lines, train, split):
        ## !!!!! THIS VARY WITH DIFFRENT DATA SET 
        ## the total number of chunk type is 11, but
        ## with 'B' 'I' tag, there should be 22, and add 'O' tag, 23.
        #if not self.label_dict.has_key(u'i-限定词'):
        #    self.label_dict[u'i-限定词'] = 124
        #if not self.label_dict.has_key(u'o-实体名'):
        #    self.label_dict[u'o-实体名'] = 125           


        sents = [] # store words
        sent = []
        tagss = [] # store lower tag
        chunk_label = []
        labels = [] # store higher tag
        label_label = []
        av_features = []
        av_feature = []
        lex_features = []
        lex_feature = []
        en_features = []
        en_feature = []

        if not train:
            self.unseen_words = {}
            self.unseen_tags = {}
            self.unseen_labels = {}
            seen_words = {}
            seen_tags = {}
            seen_labels = {}


        ## !!!!! THIS VARY WITH DIFFRENT DATA SET 
        ## unkonw_XXX's index is 1, so others begin with 2
        word_index = 2
        chunk_index = 1
        label_index = 1

        for line in lines:
            #print line
            if len(line) == 0:
                ## file end with two empty lines
                if len(sent) == 0:
                    continue
                if len(sent) > self.max_length:
                    self.max_length = len(sent)
                sents.append(sent[:])
                sent = []
                #chunk = []
                tagss.append(chunk_label[:])
                chunk_label = []
                labels.append(label_label[:])
                label_label = []

                av_features.append(av_feature[:])
                av_feature = []

                lex_features.append(lex_feature[:])
                lex_feature = []

                en_features.append(en_feature[:])
                en_feature = []                

                continue
            l = line.split(split)
            #print l
            #word_tag, chunk_tag, label_tag = l[0], l[1], l[2]
            #print int(l[5]), int(l[6]), int(l[7]), int(l[8]), int(l[9])
            try:
                word_tag, chunk_tag, lex, label_tag, en1, en2, av1, av2, av3, av4, av5 = l[0], l[1], l[2], 'NN', int(l[3]), int(l[4]), int(l[5]), int(l[6]), int(l[7]), int(l[8]), int(l[9])
            except:
                print l

            assert (en1 in [0, 1, 2, 4, 5])
            assert (en2 in [0, 1, 2, 4, 5])
            if en1 > 3:
                en1 = en1 - 1
            if en2 > 3:
                en2 = en2 - 1
            word_en_feature = [en1, 5+en2]

            word_lex_feature = 0
            if lex == 'C':
                word_lex_feature = 0
            elif lex == 'E':
                word_lex_feature = 1
            elif lex == 'O':
                word_lex_feature = 2
            elif lex == 'N':
                word_lex_feature = 3
            elif lex == 'P':
                word_lex_feature = 4

            #print av1 
            assert av1 <= 11 # 12 possiable values
            assert av2 <= 8 # 9 possiable values
            assert av3 <= 7 # 8 possible values
            assert av4 <= 6 # 7 possible values
            assert av5 <= 6 # 7 possible values
            # total 43 possible values
            word_av_feature = [0]*5
            if train:
                if not self.word_dict.has_key(word_tag):
                    self.word_dict[word_tag] = word_index
                    word_index += 1
                if not self.tag_dict.has_key(chunk_tag):
                    self.tag_dict[chunk_tag] = chunk_index
                    chunk_index += 1
                if not self.label_dict.has_key(label_tag):
                    self.label_dict[label_tag] = label_index
                    label_index += 1
            else:
                if not self.word_dict.has_key(word_tag):
                    self.word_dict[word_tag] = 1
                    self.unseen_words[word_tag] = 1
                    word_tag = '_unknown_word_'
                else:
                    seen_words[word_tag] = 1
                if not self.tag_dict.has_key(chunk_tag):
                    self.tag_dict[chunk_tag] = 1
                    self.unseen_tags[chunk_tag] = 1
                    chunk_tag = '_unknown_tag_'
                else:
                    seen_tags[chunk_tag] = 1
                if not self.label_dict.has_key(label_tag):
                    self.label_dict[label_tag] = 1
                    self.unseen_labels[label_tag] = 1
                    label_tag = '_unknown_tag_'
                else:
                    seen_labels[label_tag] = 1

            word_av_feature[0] = av1
            word_av_feature[1] = av2 + 12
            word_av_feature[2] = av3 + 21
            word_av_feature[3] = av4 + 29
            word_av_feature[4] = av5 + 36

            #print word_av_feature

            # assert sum(word_av_feature) == 5

            av_feature.append(word_av_feature[:])

            lex_feature.append(word_lex_feature)

            en_feature.append(word_en_feature[:])

            sent.append(self.word_dict[word_tag])
            chunk_label.append(self.tag_dict[chunk_tag])
            label_label.append(self.label_dict[label_tag])

        ## all numpy matrix
        self.sents = self.convertNumpy(sents)
        self.tagss = self.convertNumpy(tagss)
        self.labels = self.convertNumpy(labels)

        self.av_features = self.convertNumpy(av_features)
        print 'av', av_features[0][0]

        self.lex_features = self.convertNumpy(lex_features)
        print 'lex', lex_features[0][0]        

        self.en_features = self.convertNumpy(en_features)

        if not train:
            if len(self.unseen_words) > 0:
                print 'Word2Idx WARNING: Test ' + str(len(self.unseen_words)) + ' unknown words appear, like \'' + self.unseen_words.keys()[0].encode('utf8') + '\''
            if len(self.unseen_tags) > 0:
                print 'Word2Idx WARNING: Test ' + str(len(self.unseen_tags)) + ' unknown lower tags appear, like \'' + self.unseen_tags.keys()[0].encode('utf8') + '\''
            if len(self.unseen_labels) > 0:
                print 'Word2Idx WARNING: Test ' + str(len(self.unseen_labels)) + ' unknown higher tags, like \'' + self.unseen_labels.keys()[0].encode('utf8') + '\''

            self.seen_words_num = len(seen_words) + len(self.unseen_words)
            self.seen_tags_num = len(seen_tags) + len(self.unseen_tags)
            self.seen_labels_num = len(seen_labels) + len(self.unseen_labels)

        else:
            self.seen_words_num = len(self.word_dict) - 1
            self.seen_tags_num = len(self.tag_dict)
            self.seen_labels_num = len(self.label_dict)

    def get_data_info(self):
        print 'Data Info:'
        print ' Samples size', len(self.sents)
        print ' Sentence max length', self.max_length
        print ' Number of word types (unknown excluded)', self.seen_words_num
        print ' Number of tag types (unknown excluded)', self.seen_tags_num
        print ' Number of label types (unknown excluded)', self.seen_labels_num

    def get_data(self):
        return self.sents, self.tagss, self.labels

    def get_dicts(self):
        return self.word_dict, self.tag_dict, self.label_dict

    def save_data(self, prefix='base', train=True):
        def reverseKeyValue(d):
            _d = OrderedDict()
            for key in d.keys():
                _d[d[key]] = key
            return _d
        time_str = time.strftime("%Y-%m-%d", time.localtime()) 
        cPickle.dump((self.sents, self.tagss, self.labels, self.av_features, self.lex_features, self.en_features), open(prefix + '_data.pkl', "wb"))
        if train:
            cPickle.dump((reverseKeyValue(self.word_dict), reverseKeyValue(self.tag_dict), reverseKeyValue(self.label_dict)),\
         open(prefix + '_dict.pkl', "wb"))
        else:
            f = codecs.open(prefix + '_unseen_tokens.txt', 'w', encoding='utf-8')
            f.write('\n'.join(self.unseen_words.keys() + self.unseen_tags.keys() + self.unseen_labels.keys()))
            f.close()


if __name__ == '__main__':

    ## Train: sentence max length 52, example number 98794
    ## val sentence max length 47, example number 872
    ## test sentence max length 56, example number 1821
    ## vocabulary size 16373 (0 is not included)
    print '\nTrain'
    train = NLPCC()
    train.process('nlpcc2016_train_addfeatures_close.txt', split='\t')
    train.get_data_info()
    #words, tagss, labels = train.get_data()
    #print words
    #print tagss
    #print labels
    word_dict, tag_dict, label_dict = train.get_dicts()
    train.save_data('nlpcc2016_train')
    
    print tag_dict
    # for i in word_dict.keys():
    #     print i.encode('utf-8'), word_dict[i],
    # for i in label_dict.keys():
    #     print i.encode('utf-8'), label_dict[i], 

    print '\nDev'
    test = NLPCC(word_dict, tag_dict, label_dict)
    test.process('nlpcc2016_dev_addfeatures_close.txt', train=False, split='\t')
    test.get_data_info()
    #words, tagss, labels = test.get_data()
    #print words
    #print tagss
    #print labels
    test.save_data('nlpcc2016_dev', train=False)

    print '\nTest'
    test = NLPCC(word_dict, tag_dict, label_dict)
    test.process('nlpcc2016_test_addfeatures_close.txt', train=False, split='\t')
    test.get_data_info()
    #words, tagss, labels = test.get_data()
    #print words
    #print tagss
    #print labels
    test.save_data('nlpcc2016_test', train=False)

# Train
# Processing...
# Data Info:
#  Samples size 8999
#  Sentence max length 132
#  Number of word types (unknown excluded) 3881
#  Number of tag types (unknown excluded) 4
#  Number of label types (unknown excluded) 35
# OrderedDict([('B', 1), ('E', 2), ('S', 3), ('I', 4)])

# Dev
# Processing...
# Word2Idx WARNING: Test 85 unknown words appear, like '碁'
# Data Info:
#  Samples size 1000
#  Sentence max length 118
#  Number of word types (unknown excluded) 2488
#  Number of tag types (unknown excluded) 4
#  Number of label types (unknown excluded) 34

# Test
# Processing...
# Word2Idx WARNING: Test 269 unknown words appear, like '昀'
# Data Info:
#  Samples size 5000
#  Sentence max length 140
#  Number of word types (unknown excluded) 3622
#  Number of tag types (unknown excluded) 4
#  Number of label types (unknown excluded) 35
