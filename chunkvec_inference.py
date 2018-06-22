#coding=utf-8
import sys
import lasagne
import theano
import cPickle
import numpy
import random
import time
import codecs
import argparse
from numpy.random import RandomState
from utils import utils
from utils import generic_utils
from utils.tag_inference import TagInferenceLayer
from utils.grad import adagrad_norm, sgd_norm
from utils.CWEmbedding import CWEmbeddingLayer
from utils.CWDropout import CWDropoutLayer

parser = argparse.ArgumentParser()
parser.add_argument('--reuse', action='store_true', default=False, help='use trained mode or not (False)')
parser.add_argument('--wordvec_init', action='store_true', default=False, help='use pre-train wordvec or not (False)')
parser.add_argument('--embedding_fix', action='store_true', default=False, help='whether pre-trained wordvec be finetune')
parser.add_argument('--bio_c_flag', action='store_false', default=True, help='tag evaluate metrics choose (True)')
parser.add_argument('--batch_size', default=32, type=int, help='batch size (32)')
parser.add_argument('--max_length', default=187, type=int, help='sentence max length (140)')
parser.add_argument('--word_dim', default=100, type=int, help='word embedding dimension (50)')
parser.add_argument('--bi_dim', default=20, type=int, help='word embedding dimension (50)')
parser.add_argument('--voca_size', default=4505, type=int, help='vocabulary size (3968) include unknown, start, end')
parser.add_argument('--chunk_hidden', default=100, type=int, help='chunk LSTM hidden units (64)')
parser.add_argument('--chunk_label', default=4, type=int, help='chunk label num (4)')
parser.add_argument('--sample_use', default=-1, type=int, help='how many sample to train (-1)')
parser.add_argument('--test_sample_use', default=-1, type=int, help='how many sample to test (-1)')
parser.add_argument('--w_drop', default=0.2, type=float, help='input dropout rate, 0 for no dropout (0.2)')
parser.add_argument('--c_drop', default=0.2, type=float, help='chunk lstm dropout rate, 0 for no dropout (0.2)')
parser.add_argument('--learning_rate', default=0.2, type=float, help='learning_rate (0.2)')
parser.add_argument('--snapshot', default=-1, type=int, help='epochs a snapshot (-1)')
parser.add_argument('--num_epochs', default=10000, type=int, help='number of num_epochs (100)')
parser.add_argument('--seed', default=1, type=int, help='set seed for reproducibility (1)')
parser.add_argument('--lambda', default=1e-4, type=float, help='l2 penalty (1e-4)')
parser.add_argument('--margin_discount', default=0.2, type=float, help='margin_discount (0.1)')
parser.add_argument('--window_size', default=3, type=int, help='window_size (3)')
parser.add_argument('--test_mode', default=0, type=int, help='normal run (0), only test model (1), test model with 100 samples (2)')
parser.add_argument('--buckets', default='', type=str, help='empty for no buckets, integer list for buckets, like 0 3 4. \
                                                                                last number should equal to max_length + 1')
args = parser.parse_args()
args_hash = vars(args)

# test mode
TEST_MODE = args_hash['test_mode']
## force program into test mode
if TEST_MODE == 1:
    args_hash['wordvec_init'] = False
    #print args_hash['wordvec_init']
elif TEST_MODE == 2:
    args_hash['wordvec_init'] = False
    if args_hash['sample_use'] == -1:
        args_hash['sample_use'] = 100
    if args_hash['test_sample_use'] == -1:
        args_hash['test_sample_use'] = 100
    if args_hash['num_epochs'] == 10000:
        args_hash['num_epochs'] = 5
    if args_hash['bi_dim'] == 50:
        args_hash['bi_dim'] = 1

# if embedding_fix is set to True, wordvec_init must be True
embedding_fix = args_hash['embedding_fix']
if embedding_fix == True:
    args_hash['wordvec_init'] = True

# use trained mode or not
reuse_mode = args_hash['reuse']

# use pre-train wordvec or not
wordvec_init = args_hash['wordvec_init']

# tag evaluate metrics choose
BIO_C_FLAG = args_hash['bio_c_flag']

# batch size 
N_BATCH = args_hash['batch_size']

# sentence max length
MAX_LENGTH = args_hash['max_length']

# word embedding dimension
WORD_DIM = args_hash['word_dim']

BI_DIM = args_hash['bi_dim']

# vocabulary size
VOCA_SIZE = args_hash['voca_size']

# chunk LSTM hidden units
N_CHUNK_HIDDEN = args_hash['chunk_hidden']

# chunk label num
N_CHUNK_LABEL = args_hash['chunk_label']

#how many sample to use, -1 represent all
NUM_SAMPLE_USE = args_hash['sample_use']

NUM_TEST_SAMPLE_USE = args_hash['test_sample_use']

# chunk lstm dropout rate, 0 for no dropout
C_DROP = args_hash['c_drop']

W_DROP = args_hash['w_drop']

# learning rate
LR = args_hash['learning_rate']

# epochs a snapshot
SNAPSHOT = args_hash['snapshot']

# number of epoch
NUM_EPOCHS = args_hash['num_epochs']

# seed
SEED = args_hash['seed']

LAMBDA = args_hash['lambda']

MARGIN_DISCOUNT = args_hash['margin_discount']

WINDOW_SIZE = args_hash['window_size']

BUCKTES = []
if len(args_hash['buckets']) != 0: # buckets represents with a string, like '0 3 5', equal [0, 3), [3, 5)
    l = args_hash['buckets'].split() # a list, [3, 5]
    # max buckets size can only be equal to self.max_length (max buckets size is open number)
    if int(l[-1]) >= MAX_LENGTH + 1 and int(l[0]) == 0:
        raise ValueError('Buckets Argument Error')
    BUCKTES.append((0, int(l[0])))
    for i in range(len(l) - 1):
        if int(l[i]) >= int(l[i + 1]):
            raise ValueError('Buckets Argument Error')
        BUCKTES.append((int(l[i]), int(l[i + 1])))
    BUCKTES.append((int(l[-1]), MAX_LENGTH + 1))
else:
    BUCKTES.append((0, MAX_LENGTH + 1))

print '\nArgment Collection:', args_hash, '\n'

TRAIN_DATA_FILE = "nlpcc2016_train_data.pkl"
TEST_DATA_FILE = "nlpcc2016_test_data.pkl"
VAL_DATA_FILE = "nlpcc2016_dev_data.pkl"
#WORD_VEC_FILE = "/home/zqr/code/data_chunkvec/chunk/nlpcc_wordvec.pkl"
DICT_FILE = "nlpcc2016_train_dict.pkl"
TEST_GS_FILE = 'nlpcc2016_test.txt'
VAL_GS_FILE = 'nlpcc2016_dev.txt'
DICTIONARY='nlpcc_dictionary.txt'

#################################################################################################

# set seed for reproducibility
lasagne.random.set_rng(RandomState(SEED))

def load_data():
    
    train_sents, train_tagss, train_label, train_av, train_lex, train_en = cPickle.load(open(TRAIN_DATA_FILE, "rb"))
    val_sents, val_tagss, val_label, val_av, val_lex, val_en = cPickle.load(open(VAL_DATA_FILE, "rb"))
    test_sents, test_tagss, test_label, test_av, test_lex, test_en = cPickle.load(open(TEST_DATA_FILE, "rb"))
    word_dict, tag_dict, label_dict = cPickle.load(open(DICT_FILE, "rb"))

    train_sents = train_sents[:NUM_SAMPLE_USE]
    train_label = train_label[:NUM_SAMPLE_USE]
    train_tagss = train_tagss[:NUM_SAMPLE_USE]
    train_av = train_av[:NUM_SAMPLE_USE]
    train_lex = train_lex[:NUM_SAMPLE_USE]
    train_en = train_en[:NUM_SAMPLE_USE]

    val_sents = val_sents[:NUM_TEST_SAMPLE_USE]
    val_label = val_label[:NUM_TEST_SAMPLE_USE]
    val_tagss = val_tagss[:NUM_TEST_SAMPLE_USE]
    val_av = val_av[:NUM_SAMPLE_USE]
    val_lex = val_lex[:NUM_SAMPLE_USE]
    val_en = val_en[:NUM_SAMPLE_USE]

    test_sents = test_sents[:NUM_TEST_SAMPLE_USE]
    test_label = test_label[:NUM_TEST_SAMPLE_USE]
    test_tagss = test_tagss[:NUM_TEST_SAMPLE_USE]
    test_av = test_av[:NUM_SAMPLE_USE]
    test_lex = test_lex[:NUM_SAMPLE_USE]
    test_en = test_en[:NUM_SAMPLE_USE]

    return (train_sents, train_label, train_tagss, train_av, train_lex, train_en,
            val_sents, val_label, val_tagss, val_av, val_lex, val_en,
             test_sents, test_label, test_tagss, test_av, test_lex, test_en,
              word_dict, tag_dict, label_dict)


def contextwin_bigram(m, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    
    out = []
    bi = []
    
    def bigram(s):
        return [(s[i] * VOCA_SIZE + s[i + 1]) for i in range(len(s) - 1)]
            
    for l in m:
        l = list(l)
        
        #print l
        lpadded = win // 2 * [VOCA_SIZE - 2] + l + win // 2 * [VOCA_SIZE - 1]
        #print lpadded
        l_out = [lpadded[i:(i + win)] for i in range(len(l))]
        l_bi = [bigram(lpadded[i:(i + win)]) for i in range(len(l))]
        
        assert len(l_out) == len(l)
        out.append(l_out)
        bi.append(l_bi)
        
    return utils.convertNumpy(out), utils.convertNumpy(bi)

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# inputs: (example num, max length), masks: (example num, max length)
# sent_targets: (example num, sentiment label num)
# chunk_targets: (example num, chunk label num)
    ## batch iterator
def iterate_minibatches(inputs, sent_targets, chunk_targets, train_av, train_lex, train_en, batchsize, shuffle=False):

    assert len(inputs) == len(sent_targets)
    assert len(inputs) == len(chunk_targets)

    def gene_mask(X):
        return numpy.int32(numpy.ones_like(X) * (1 - numpy.equal(X, 0)))

    index_set = [[] for _ in BUCKTES]
    for i in range(len(inputs)):
        x = inputs[i]
        for bucket_id, (min_size, max_size) in enumerate(BUCKTES):
            if len(x) >= min_size and len(x) < max_size:
                index_set[bucket_id].append(i)
                break

    if index_set and len(index_set) > 1: # user set multi buckets
        for i in range(len(index_set)):
            bucket_max_length = BUCKTES[i][1] - 1
            index_bucket = index_set[i]
            if len(index_bucket) == 0: # empty buckets
                #print 'empty'
                continue
            #print index_bucket
            index_bucket_shuffled = index_bucket[:]
            if shuffle:
                lasagne.random.get_rng().shuffle(index_bucket_shuffled)
            for start_idx in range(0, len(index_bucket), batchsize):
                if shuffle:
                    excerpt = index_bucket_shuffled[start_idx:len(index_bucket) \
                    if (start_idx + batchsize > len(index_bucket)) else start_idx + batchsize]
                else:
                    excerpt = index_bucket[start_idx:len(index_bucket) \
                    if (start_idx + batchsize > len(index_bucket)) else start_idx + batchsize]

                sents_one_batch = utils.pad_sequences([inputs[j] for j in excerpt], bucket_max_length)
                masks_one_batch = gene_mask(sents_one_batch)
                sentscw_one_batch = contextwin(sents_one_batch, WINDOW_SIZE)
                sent_targets_one_batch = utils.pad_sequences([sent_targets[j] for j in excerpt], 
                                                            bucket_max_length)
                chunk_targets_one_batch = utils.pad_sequences([chunk_targets[j] for j in excerpt], 
                                                                bucket_max_length)

                yield sentscw_one_batch, masks_one_batch, sent_targets_one_batch, chunk_targets_one_batch

    else: # only one default bucket: (0, max_length) or None buckets at all
        if shuffle:
            indices = numpy.arange(len(inputs))
            lasagne.random.get_rng().shuffle(indices)
            #print indices
        for start_idx in range(0, len(inputs), batchsize):
            if shuffle:
                excerpt = indices[start_idx:len(inputs) \
                if (start_idx + batchsize > len(inputs)) else start_idx + batchsize]
            else:
                excerpt = range(start_idx, len(inputs) \
                    if (start_idx + batchsize > len(inputs)) else start_idx + batchsize)

            sents_one_batch = utils.pad_sequences([inputs[j] for j in excerpt], MAX_LENGTH)
            masks_one_batch = gene_mask(sents_one_batch)
            sentscw_one_batch, bigram_one_batch = contextwin_bigram(sents_one_batch, WINDOW_SIZE)
            train_av_one_batch = [utils.pad_matrix(train_av[j], sent_maxlen=MAX_LENGTH, feature_dim=5) for j in excerpt]
            train_lex_one_batch = utils.pad_sequences([train_lex[j] for j in excerpt], MAX_LENGTH)
            train_en_one_batch = [utils.pad_matrix(train_en[j], sent_maxlen=MAX_LENGTH, feature_dim=2) for j in excerpt]
            sent_targets_one_batch = utils.pad_sequences([sent_targets[j] for j in excerpt], MAX_LENGTH)
            chunk_targets_one_batch = utils.pad_sequences([chunk_targets[j] for j in excerpt], MAX_LENGTH)

            #yield sentscw_one_batch, masks_one_batch, masks_seg_one_batch, sent_targets_one_batch, chunk_targets_one_batch
            yield sentscw_one_batch, bigram_one_batch, train_av_one_batch, train_lex_one_batch, train_en_one_batch, masks_one_batch, \
            sent_targets_one_batch, chunk_targets_one_batch


def build_model(sents=None, bigrams=None, av_features=None, lex_features=None, en_features=None, masks=None, chunk_labels=None, embedding_vec=None):

    # word embedding layer, output shape: (batch size, max length, word dim)
    emb_layer_flatten = None
    bi_emb_layer_flatten = None
    if embedding_fix == True:
        embeddings = theano.shared(embedding_vec, 'embeddings')
        emb_layer = lasagne.layers.InputLayer(shape=(None, None, WORD_DIM),
                                                input_var=embeddings[sents])
    else:   
        # input layer, input sentence
        input_layer = lasagne.layers.InputLayer(shape=(None, None, WINDOW_SIZE),
                                            input_var=sents)
        input_layer_dp = CWDropoutLayer(input_layer, p=W_DROP, rescale=False)

        bi_input_layer = lasagne.layers.InputLayer(shape=(None, None, WINDOW_SIZE - 1),
                                            input_var=bigrams)
        bi_input_layer_dp = CWDropoutLayer(bi_input_layer, p=W_DROP, rescale=False)

        av_input_layer = lasagne.layers.InputLayer(shape=(None, None, 5),
                                            input_var=av_features)

        lex_input_layer = lasagne.layers.InputLayer(shape=(None, None),
                                            input_var=lex_features) 

        en_input_layer = lasagne.layers.InputLayer(shape=(None, None, 2),
                                            input_var=en_features)                                                   

        # W init with Normal function(std=0.01, mean=0)
        if wordvec_init == True:
            #emb_layer = CWEmbeddingLayer(input_layer_dp, W=embedding_vec, input_size=VOCA_SIZE, output_size=WORD_DIM, window_size=WINDOW_SIZE)
            emb_layer = lasagne.layers.EmbeddingLayer(input_layer_dp, W=embedding_vec, 
                input_size=VOCA_SIZE, output_size=WORD_DIM)
            emb_layer_flatten = lasagne.layers.FlattenLayer(emb_layer, outdim=3)
            bi_emb_layer = lasagne.layers.EmbeddingLayer(bi_input_layer_dp, W=bi_embedding_vec, 
                input_size=VOCA_SIZE*VOCA_SIZE, output_size=BI_DIM)
            bi_emb_layer_flatten = lasagne.layers.FlattenLayer(bi_emb_layer, outdim=3)
        else:
            #emb_layer = CWEmbeddingLayer(input_layer_dp, input_size=VOCA_SIZE, output_size=WORD_DIM, window_size=WINDOW_SIZE)
            emb_layer = lasagne.layers.EmbeddingLayer(input_layer_dp, input_size=VOCA_SIZE, 
                output_size=WORD_DIM)
            emb_layer_flatten = lasagne.layers.FlattenLayer(emb_layer, outdim=3)
            bi_emb_layer = lasagne.layers.EmbeddingLayer(bi_input_layer_dp, 
                input_size=VOCA_SIZE*VOCA_SIZE, output_size=BI_DIM)
            bi_emb_layer_flatten = lasagne.layers.FlattenLayer(bi_emb_layer, outdim=3)
            av_emb_layer = lasagne.layers.EmbeddingLayer(av_input_layer, 
                input_size=43, output_size=30)
            av_emb_layer_flatten = lasagne.layers.FlattenLayer(av_emb_layer, outdim=3)
            lex_emb_layer = lasagne.layers.EmbeddingLayer(lex_input_layer, 
                input_size=5, output_size=30) 
            en_emb_layer = lasagne.layers.EmbeddingLayer(en_input_layer, 
                input_size=10, output_size=30)
            en_emb_layer_flatten = lasagne.layers.FlattenLayer(en_emb_layer, outdim=3)                           

    emb_layer = lasagne.layers.ConcatLayer([emb_layer_flatten, bi_emb_layer_flatten, av_emb_layer_flatten, lex_emb_layer, en_emb_layer_flatten], axis=2)

    ## get batch sentence max length
    batch_max_length = lasagne.layers.get_output(input_layer).shape[1]

    #print 'emb_layer output shape', lasagne.layers.get_output_shape(emb_layer)

    # mask matrix for LSTM layers 
    mask_layer = lasagne.layers.InputLayer(shape=(None, None),
                                           input_var=masks)

    # forward LSTM layer, processing sentence, output shape: (batch size, max length, chunk dim)
    chunk_lstm = lasagne.layers.LSTMLayer(
        emb_layer, N_CHUNK_HIDDEN,
        mask_input=mask_layer,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=False,
        precompute_input=True,
        name='chunk_f')

    #print 'chunk lstm', lasagne.layers.get_output_shape(chunk_lstm)

    chunk_lstm_dp = None
    if C_DROP > 0:
        chunk_lstm_dp = lasagne.layers.DropoutLayer(chunk_lstm, p=C_DROP)
    else:
        chunk_lstm_dp = chunk_lstm

    # In order to connect a recurrent layer to a dense layer, we need to
    # flatten the first two dimensions (our "sample dimensions"); this will
    # cause each time step of each sequence to be processed independently
    # finally chunk label output shape (batch size, max length, chunk label num)
    chunk_shp = lasagne.layers.ReshapeLayer(chunk_lstm_dp, (-1, N_CHUNK_HIDDEN))
    #print 'chunk_shp', lasagne.layers.get_output_shape(chunk_shp)
    chunk_dense = lasagne.layers.DenseLayer(chunk_shp, num_units=N_CHUNK_LABEL,
                             nonlinearity=lasagne.nonlinearities.softmax)
    #print 'chunk_dense', lasagne.layers.get_output_shape(chunk_dense)
    chunk_reshp = lasagne.layers.ReshapeLayer(chunk_dense, (-1, batch_max_length, 
        N_CHUNK_LABEL))
    #print 'chunk_out', lasagne.layers.get_output_shape(chunk_out)

    ## !!! THIS PARAMETERIZED TAG INFERENCE LAYER IS ONLY SUITABLE FOR CWS !!!
    ## OUR LABELS ARE (0, B), (1, E), (2, S), (3, I)
    init_loss = numpy.asarray([0, -numpy.inf, 0, -numpy.inf])
    tran_loss = numpy.asarray([[-numpy.inf, 0, -numpy.inf, 0],
                               [0, -numpy.inf, 0, -numpy.inf], 
                               [0, -numpy.inf, 0, -numpy.inf],
                               [-numpy.inf, 0, -numpy.inf, 0]])
    halt_loss = numpy.asarray([-numpy.inf, 0, 0, -numpy.inf])

    chunk_out = TagInferenceLayer(chunk_reshp,
                                  N_CHUNK_LABEL,
                                  MARGIN_DISCOUNT,
                                  init_loss=init_loss,
                                  tran_loss=tran_loss,
                                  halt_loss=halt_loss,
                                  gs_t=chunk_labels,
                                  mask_input=masks,
                                  name='tag_inference')     
 
    return chunk_out 

def main():

    # word vector
    embedding_vec = None
    ## model testing doesn't need data
    if TEST_MODE != 1:
        if TEST_MODE == 2:
            print 'INTO TEST_MODE 2'
        ## load the dataset
        print 'Load data...'
        X_train, y_train_sent, y_train_chunk, train_av, train_lex, train_en,\
          X_val, y_val_sent, y_val_chunk, val_av, val_lex, val_en,\
            X_test, y_test_sent, y_test_chunk, test_av, test_lex, test_en,\
              word_dict, tag_dict, label_dict = load_data()

        # read wordvec file
        if wordvec_init == True:
            embedding_vec = numpy.empty((0, WORD_DIM), float)
            wordvec_dict = cPickle.load(open(WORD_VEC_FILE, 'rb'))
            for i in range(VOCA_SIZE):
                if wordvec_dict.has_key(i):
                    embedding_vec = numpy.append(embedding_vec, 
                        wordvec_dict[i].reshape(1, WORD_DIM), axis=0)
                else:
                    embedding_vec = numpy.append(embedding_vec, 
                        lasagne.random.get_rng().normal(0.0, 0.01, size=(1, WORD_DIM)), axis=0)
    #print embedding_vec.shape, type(embedding_vec)
    #print embedding_vec[0][:10], len(embedding_vec[0]) 
        print 'Ok.'
    else:
        print 'INTO TEST_MODE 1'    

    ### model inputs
    sents = theano.tensor.itensor3('sents')
    bigrams = theano.tensor.itensor3('bigrams')
    masks = theano.tensor.imatrix('masks')
    av_features = theano.tensor.itensor3('av')
    lex_features = theano.tensor.imatrix('lex')
    en_features = theano.tensor.itensor3('en')
    #masks_seg = theano.tensor.imatrix('masks_seg')
    lr = theano.tensor.fscalar('lr')

    ### model target outputs    
    # chunk label target
    chunk_labels = theano.tensor.imatrix()
    ### model target outputs
    def expandMatrix(X, dim=None, exflag=True):
        ## exflag: has padding or not.
        # (batch size, max length) -> (batch size, max length, dim)
        _eye = None
        if exflag:
            _x = theano.tensor.eye(dim)
            _y = theano.tensor.zeros((1, dim))
            _eye = theano.tensor.concatenate([_y, _x], axis=0)
        else:
            _eye = theano.tensor.eye(dim)
        return theano.tensor.cast(_eye[X], dtype='int32')
    chunk_targets = expandMatrix(chunk_labels, N_CHUNK_LABEL)

    ## build model
    print 'Build model...'
    
    # the model has two outputs
    chunk_out = build_model(sents, bigrams, av_features, lex_features, en_features, masks, chunk_labels, embedding_vec)

    # whether or not use trained model
    if reuse_mode == True:
        with numpy.load('model_best_c_test.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(chunk_out, param_values) 
    
    # (batch size, )
    chunk_label_mass = lasagne.layers.get_output(chunk_out)
    chunk_loss = theano.tensor.mean(chunk_label_mass[0] / theano.tensor.sum(masks, axis=1))
    chunk_label_chain = chunk_label_mass[1] * masks

   # l2 penalty
    loss_penalty = lasagne.regularization.regularize_layer_params(chunk_out, lasagne.regularization.l2) * LAMBDA

    loss = chunk_loss + loss_penalty

    all_params = lasagne.layers.get_all_params(chunk_out, trainable=True)
    #print all_params

    ## set constraints for Tag Inference Layer
    tag_inference_layer_params = chunk_out.get_params()
    init_tran = tag_inference_layer_params[0]
    tran = tag_inference_layer_params[1]
    halt_tran = tag_inference_layer_params[2]

    def l1_unit_norm(p):
        epsilon = 10e-8
        p = p * theano.tensor.cast(p >= 0., 'float64')
        return p / (epsilon + theano.tensor.sum(p, axis=-1, keepdims=True))

    constraints = {init_tran:l1_unit_norm, tran:l1_unit_norm, halt_tran:l1_unit_norm}


    updates = adagrad_norm(loss, all_params, learning_rate=lr, constraints=constraints)   

    ## for validation and test
    chunk_pred_label_mass = lasagne.layers.get_output(chunk_out, deterministic=True)
    chunk_pred_loss = theano.tensor.mean(chunk_pred_label_mass[0] / theano.tensor.sum(masks, axis=1))
    chunk_pred_label_chain = chunk_pred_label_mass[1] * masks

    val_loss = chunk_pred_loss
 
    # for train
    train_fn = theano.function([sents, bigrams, av_features, lex_features, en_features, masks, chunk_labels, lr],
                               loss,
                               updates=updates)
                               # loss, updates=updates)
    # validation or test
    val_fn = theano.function([sents, bigrams, av_features, lex_features, en_features, masks, chunk_labels],
                             [val_loss, chunk_pred_label_chain])
    
    if TEST_MODE == 1:
        print 'MODEL BUILDING PASS.'
        sys.exit()

    print 'Ok.'

    best_val_c_f1 = 0
    best_val = 0
    best_test_c_f1 = 0

    data = {'best_val_c_f1':[], 'best_val':[], 'best_test_c_f1':[]}

    lr_decayed = numpy.float32(LR)

    train_losses = []   

    # Finally, launch the training loop.
    print "Starting training..."
    for epoch in range(NUM_EPOCHS):
        print 'Epoch', epoch

        progbar = generic_utils.Progbar(X_train.shape[0])

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batchs = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train_sent, y_train_chunk, train_av, train_lex, train_en, N_BATCH, shuffle=True):
            inputs, bigrams, av_features, lex_features, en_features, masks, sentiment_targets, chunk_targets = batch
            err = train_fn(inputs, bigrams, av_features, lex_features, en_features, masks, chunk_targets, lr_decayed)
            train_err += err
            train_batchs += 1

            progbar.add(inputs.shape[0], values=[('train loss', err)])
        train_loss = train_err / train_batchs
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(train_losses) > 3 and train_loss > max(train_losses[-3:]):
            lr_decayed = numpy.float32(lr_decayed * 0.5)

        train_losses.append(train_loss)

        val_cf1 = 0
        # And a full pass over the val data: 
        inputs = utils.pad_sequences(X_val, MAX_LENGTH)
        masks = numpy.int32(numpy.ones_like(inputs) * (1 - numpy.equal(inputs, 0)))
        inputscw, bigrams = contextwin_bigram(inputs, WINDOW_SIZE)
        chunk_targets = utils.pad_sequences(y_val_chunk, MAX_LENGTH)
        val_av_batch = [utils.pad_matrix(val_av[j], sent_maxlen=MAX_LENGTH, feature_dim=5) for j in range(len(val_av))]
        val_lex_batch = utils.pad_sequences(val_lex, MAX_LENGTH)
        val_en_batch = [utils.pad_matrix(val_en[j], sent_maxlen=MAX_LENGTH, feature_dim=2) for j in range(len(val_en))]
        val_err, val_chunk_label = val_fn(inputscw, bigrams, val_av_batch, val_lex_batch, val_en_batch, masks, chunk_targets)

        if BIO_C_FLAG:
            c_res_val = utils.cwsEalve(inputs, chunk_targets, val_chunk_label,
                word_dict, tag_dict, VAL_GS_FILE, DICTIONARY, False)
            val_cf1 = c_res_val

        if best_val <= (val_cf1):
            best_val = val_cf1
            best_val_c_f1 = val_cf1

            test_cf1 = 0
            # And a full pass over the test data:
            inputs = utils.pad_sequences(X_test, MAX_LENGTH)
            masks = numpy.int32(numpy.ones_like(inputs) * (1 - numpy.equal(inputs, 0)))
            inputscw, bigrams = contextwin_bigram(inputs, WINDOW_SIZE)
            chunk_targets = utils.pad_sequences(y_test_chunk, MAX_LENGTH)
            test_av_batch = [utils.pad_matrix(test_av[j], sent_maxlen=MAX_LENGTH, feature_dim=5) for j in range(len(test_av))]
            test_lex_batch = utils.pad_sequences(test_lex, MAX_LENGTH)
            test_en_batch = [utils.pad_matrix(test_en[j], sent_maxlen=MAX_LENGTH, feature_dim=2) for j in range(len(test_en))]
            test_err, test_chunk_label = val_fn(inputscw, bigrams, test_av_batch, test_lex_batch, test_en_batch, masks, chunk_targets)

            if BIO_C_FLAG:
                # c_res_val = utils.cwsEalve(inputs, chunk_targets, test_chunk_label,
                #     word_dict, tag_dict, TEST_GS_FILE, DICTIONARY, False)
                utils.save_test(inputs, chunk_targets, test_chunk_label,
                     word_dict, tag_dict, TEST_GS_FILE, DICTIONARY, False)
                #best_test_c_f1 = c_res_val

            #numpy.savez('model_best_c_test.npz', *lasagne.layers.get_all_param_values(chunk_out))
            # utils.id2original(inputs, chunk_targets,
            #     utils.pad_sequences(y_test_sent, MAX_LENGTH, value=1), test_chunk_label,
            #     utils.pad_sequences(y_test_sent, MAX_LENGTH, value=1),
            #         word_dict=word_dict, tag_dict=tag_dict, label_dict=label_dict,
            #             output_file='test_c_result.txt')

        # Then we print the results for this epoch:
        print "val: %.4f c_f1: %.4f best: %.4f test: %.4f c_f1: %.4f" \
        %(val_err, val_cf1, best_val, test_err, best_test_c_f1)

        data['best_val_c_f1'].append(best_val_c_f1)
        data['best_val'].append(best_val)
        data['best_test_c_f1'].append(best_test_c_f1)
        data['args'] = args_hash
        cPickle.dump(data, open('result_data.pkl', 'wb'))

###################
#  main function  #
###################

main()



