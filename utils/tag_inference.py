import numpy as np
import theano.tensor as T
import theano
import lasagne


class TagInferenceLayer(lasagne.layers.Layer):
    """
    Added by Chanrom, 4.24.2016
    Modified by Chanrom 6.14.2016 
    lasagne.layers.TagInferenceLayer(incoming, tran_t=lasagne.init.GlorotUniform(),
    gs_t=None, mask_input=None, **kwargs)

    A tag inference layer

    Parameters
    ----------
    incoming : a :class:'Layer' instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_labels : int
        The number of labels 
        
    init_t : Theano shared variable, expression, numpy array or callable or 'None'
        Initial value, expression or initializer for the initial transition matrix for tags.
        This should be a matrix with shape '(num_label, )'.

    tran_t : Theano shared variable, expression, numpy array or callable or 'None'
        Initial value, expression or initializer for the transition matrix for tags.
        This should be a matrix with shape '(num_label, num_label)'.

    halt_t : Theano shared variable, expression, numpy array or callable or 'None'
        Initial value, expression or initializer for the end tags transition matrix.
        This should be a matrix with shape '(num_label, )'.

    gs_t : Theano shared variable, expression, numpy array, callable or 'None'
        Golden standard label sequence for caculating score of gs sequence.
        This should be a matrix with shape '(example num, max sequence length)'.
        LABEL INDEX SHOULD BE 0-BASED.

    init_loss : Numpy array or 'None'
        Transition rules when tag transfering, working for <start> to user defined labels.

    tran_loss : Numpy array or 'None'
        Transition rules when tag transfering, working for user defined labels.

    halt_loss : Numpy array or 'None'
        Transition rules when tag transfering, working for user defined labels to <end> labels.        

    mask_input : Theano shared variable, expression, numpy array, callable or 'None'
        mask matrix, this should be a matrix with shape '(example num, max sequence length)'
    """
    def __init__(self, incoming, num_labels, margin_discount, init_t=lasagne.init.Uniform(),
        tran_t=lasagne.init.Uniform(), halt_t=lasagne.init.Uniform(), init_loss=None,
        tran_loss=None, halt_loss=None, gs_t=None, mask_input=None, **kwargs):
        super(TagInferenceLayer, self).__init__(incoming, **kwargs)
        self.init = self.add_param(init_t, (num_labels, ), name="init_t")
        self.tran = self.add_param(tran_t, (num_labels, num_labels), name="tran_t")
        self.halt = self.add_param(halt_t, (num_labels, ), name='halt_t')
        self.gs = gs_t
        self.num_labels = num_labels
        self.margin_discount = margin_discount
        self.init_loss = init_loss
        self.tran_loss = tran_loss
        self.halt_loss = halt_loss
        if not mask_input:
            self.masks = np.ones((incoming.shape[0], incoming.shape[1]))
        else:
            self.masks = mask_input
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    # [a, b, c, d] -> [[a, b, c, d], [a, b, c, d], [a, b, c, d], [a, b, c, d]]
    def vector2matrix(self, x):
        ones = T.ones_like(T.reshape(x, (-1, 1)))
        return T.reshape(x, (1, -1)) * ones

    # X, shape: (batch size, ), Y, shape: (batch size, time steps - 1)
    # return Tensor shape: (batch size, time step), time step is reverse
    # we throw away start label predict
    def aggregateTensor(self, X, Y):
        _X = X.dimshuffle(0, 'x')
        return theano.tensor.concatenate([_X, Y], axis=1)   


    def score_one_step(self, pred_t, gs_t, mask_t, score_tm1, index_tm1, init_tran, tran, halt_tran, deterministic, init_flag):
        # pred_t: (batch_size, label_num)
        # gs_t: (batch_size, )
        # mask_t: (batch_size, )
        # score_tm1: (batch_size, label_num)
        # index_tm1: (batch_size, label_num)
        # gs_score_tm1: (batch_size, )
        # gs_tm1: (batch_size, )
        # init_tran: (label_num, )
        # tran: (label_num, label_num)
        # deterministic: tensor constant, 0 or 1
        # init_flag: tensor constant, 0 or 1


        def scan_example(pred_t_e, gs_t_e, mask_t_e, score_tm1_e, index_tm1_e, init_tran, tran, halt_tran, deterministic, init_flag):
            # pred_t_e: (label_num, )
            # gs_t_e: (1, )
            # mask_t_e: (1, )
            # score_tm1_e: (label_num, )
            # index_tm1_e: (label_num, )
            # gs_score_tm1_e: (1, )
            # gs_tm1_e: (1, )
            # init_tran: (label_num, )
            # tran: (label_num, label_num)
            # deterministic: tensor constant, 0 or 1
            # init_flag: tensor constant, 0 or 1

            label_num = pred_t_e.shape[0]

            score_t_e = None
            index_t_e = None

            label_num = pred_t_e.shape[0]

            ## !!! THIS TAG INFERENCE LAYER IS ONLY SUITABLE FOR CWS !!!
            ## OUR LABELS ARE (0, B), (1, E), (2, S), (3, I)
            if init_flag == T.constant(1): # initial step
                # print 'scan_example init_step'
                # we only have one label <start>, other paths are invalid
                # tran_loss = np.asarray([0, -np.inf, 0, -np.inf])
                tran_loss = self.init_loss
                index_t_e = T.cast(T.zeros((label_num,))-1, 'int64') # actually it doesn't matter
                score_t_e = pred_t_e + init_tran + tran_loss \
                 +(1 - deterministic) * self.margin_discount * (T.ones((label_num,)) - T.eye(label_num)[gs_t_e])
            elif init_flag == T.constant(0):
                # tran_loss = np.asarray([[-np.inf, 0, -np.inf, 0],
                #                         [0, -np.inf, 0, -np.inf], 
                #                         [0, -np.inf, 0, -np.inf],
                #                         [-np.inf, 0, -np.inf, 0]])
                tran_loss = self.tran_loss

                all_score_t_e = self.vector2matrix(pred_t_e) + tran + self.vector2matrix(score_tm1_e).T  + tran_loss\
                 + (1 - deterministic) * self.margin_discount * (T.ones((label_num,)) - T.eye(label_num)[gs_t_e])
                # print all_score_t_e.eval()
                # y = theano.printing.Print('\nall_score_t_e\n')(all_score_t_e)
                # x = all_score_t_e.eval({pred_t_e:[0.1, 0.5, 0.3, 0.1], score_tm1_e:[1.1, -np.inf, 0.8, -np.inf], label_num:4, gs_t_e:1})
                # print x
                # print np.max(x, axis=0)
                score_t_e = T.max(all_score_t_e, axis=0)
                index_t_e = T.argmax(all_score_t_e, axis=0)
            else:
                # tran_loss = np.asarray([-np.inf, 0, 0, -np.inf])
                tran_loss = self.halt_loss
                score_t_e = score_tm1_e + halt_tran + tran_loss
                index_t_e = T.arange(label_num)

            # if current mask bit is 0, keep score unchanged
            score_t_e = T.switch(T.eq(mask_t_e, 1), score_t_e, score_tm1_e)

            # if current mask bit is 0, keep tag unchanged, that means B -> B -> B......
            index_t_e = T.cast(index_t_e * mask_t_e + T.arange(label_num) * (1 - mask_t_e), 'int64')

            return score_t_e, index_t_e

        # return shape: (batch size, label num)...
        score_t, index_t = theano.scan(fn=scan_example,
                                        sequences=[pred_t, gs_t, mask_t, score_tm1, index_tm1],
                                        non_sequences=[init_tran, tran, halt_tran, deterministic, init_flag])[0]

        return score_t, index_t       


    def get_output_for(self, input, deterministic=False):

        if deterministic:
            deterministic_flag = T.constant(1)
        else:
            deterministic_flag = T.constant(0)
        # deterministic_flag = T.constant(0)

        batch_size = input.shape[0]
        time_steps = input.shape[1]
        label_num = self.num_labels

        ## the start state to first label
        pred_t1 = input[:, 0] # shape: (batch size, label num)
        gs_t1 = self.gs[:, 0] - 1
        mask_t1 = self.masks[:, 0]

        score_t0 = T.zeros((batch_size, label_num))
        index_t0 = T.zeros((batch_size, label_num), dtype='int64')

        init_flag = T.constant(1)
        # return shape: (batch size, label num), (batch size, label num)
        score_t1, index_t1 = self.score_one_step(pred_t1, gs_t1,
            mask_t1, score_t0, index_t0, self.init, self.tran, self.halt, deterministic_flag, init_flag)

        pred = input.dimshuffle(1, 0, 2)
        gs = self.gs.dimshuffle(1, 0)
        mask = self.masks.dimshuffle(1, 0)
        init_flag = T.constant(0)

        # return shape: (time steps - 1, batch size, label num) ..., (time steps - 1, batch size)
        step_scores, step_indexs = theano.scan(fn=self.score_one_step,
                                               outputs_info=[score_t1, index_t1],
                                               sequences=[pred[1:], gs[1:]-1, mask[1:]],
                                               non_sequences=[self.init, self.tran, self.halt, deterministic_flag, init_flag])[0]

        ## then end state for end label
        pred_tn = T.zeros((batch_size, label_num))
        gs_tn = T.zeros((batch_size, label_num))
        mask_tn = T.ones((batch_size, label_num))
        init_flag = 2
        # return shape: (batch size, label num), (batch size, label num)
        halt_score, halt_index = self.score_one_step(pred_tn, gs_tn, mask_tn, step_scores[-1], step_indexs[-1],
            self.init, self.tran, self.halt, deterministic_flag, init_flag)

        # shape: (batch size, )
        last_step_max_index = T.argmax(halt_score, axis=-1)

        def track_one_step(index_t, max_index_t):
            # example_indexs shape: (batch size, label num)
            # step_max_index shape: (batch size, )
            def scan_example(index_t_e, max_index_t_e):
                max_index_tm1_e = index_t_e[max_index_t_e]
                return max_index_tm1_e
            # return shape: (batch size, )
            max_index_tm1 = theano.scan(fn=scan_example,
                                              sequences=[index_t, max_index_t])[0]
            return max_index_tm1

        # return shape: (time steps - 1, batch size)
        index_chain = theano.scan(fn=track_one_step,
                                  sequences=step_indexs,
                                  outputs_info=last_step_max_index,
                                  go_backwards=True)[0]
        # return shape: (batch size, time steps - 1)
        index_chain = index_chain.dimshuffle(1, 0)

        # shape: (batch size, time steps)
        index_chain_reverse = self.aggregateTensor(last_step_max_index, index_chain)

        # add 1 for label index (which index from 1)
        # return shape: (batch size, time steps)
        index_chain = (index_chain_reverse + T.ones_like(index_chain_reverse))[:, ::-1]


        def one_step_cost(step_index, pred_t, gs_t, index_chain_t, mask_t, cost_tm1, gs_tm1, index_chain_tm1, init_tran, tran):
            # step_index: (1,)
            # pred_t: (batch size, label num)
            # gs_t_e: (batch size, )
            # index_chain_t: (batch size, )
            # mask_t: (batch size, )
            # cost_tm1: (batch size, )
            # gs_tm1: (batch size, )
            # index_chain_tm1: (batch size, )


            def scan_example(pred_t_e, gs_t_e, index_chain_t_e, mask_t_e, cost_tm1_e, gs_tm1_e, index_chain_tm1_e, step_index, init_tran, tran):
                # pred_t_e: (label num, )
                # gs_t_e: (1, )
                # index_chain_t_e: (1, )
                # mask_t_e: (1, )
                # gs_tm1_e: (1, )
                # index_chain_tm1_e: (1, )
                # init_tran: (label num, )
                # tran: (label num, label num)

                delta_loss = theano.ifelse.ifelse(T.eq(gs_t_e, index_chain_t_e),
                    np.float64(0.0), np.float64(self.margin_discount))

                cost_t_e = None
                cost_t_e = theano.ifelse.ifelse(T.eq(step_index, 0),
                    pred_t_e[index_chain_t_e] + init_tran[index_chain_t_e] - pred_t_e[(gs_t_e)] - init_tran[gs_t_e] + delta_loss,
                    pred_t_e[index_chain_t_e] + tran[index_chain_tm1_e][index_chain_t_e] - pred_t_e[gs_t_e] - tran[gs_tm1_e][gs_t_e] + delta_loss)

                cost_t_e = cost_t_e * mask_t_e

                # return shape: (1, )
                return cost_t_e, gs_t_e, index_chain_t_e

            # return shape: (batch size, )...
            cost_t, _, _ = theano.scan(fn=scan_example,
                                sequences=[pred_t, gs_t, index_chain_t, mask_t, cost_tm1, gs_tm1, index_chain_tm1],
                                non_sequences=[step_index, init_tran, tran])[0]

            # return shape: (batch size, )...
            return cost_t, gs_t, index_chain_t


        # return shape: (time steps, batch size)
        index_chain_sff = index_chain.dimshuffle(1, 0)
        gs_t0 = T.zeros((batch_size, ), dtype='int32')
        cost_t0 = T.zeros((batch_size, ), dtype='float64')
        index_chain_t0 = T.zeros((batch_size, ), dtype='int64')

        # return shape: (time steps, batch size)
        # print (gs-1).eval()
        # print (index_chain_sff-1).eval()
        steps_cost, _, _ = theano.scan(fn=one_step_cost,
                                 outputs_info=[cost_t0, gs_t0, index_chain_t0],
                                 sequences=[T.arange(time_steps), pred, gs-1, index_chain_sff-1, mask],
                                 non_sequences=[self.init, self.tran])[0]

        # return shape: (batch size, )
        cost = T.sum(steps_cost.dimshuffle(1, 0), axis=-1)

        #print 'cost', cost.eval()
        # return shape: (batch size, ), (batch size, time steps)
        return T.nnet.relu(cost), index_chain

if __name__=='__main__':

    pred_v = np.array([[[0.2, 0.3, 0.5],
                        [0.4, 0.3, 0.3],
                        [0.7, 0.2, 0.1]],
                       [[0.5, 0.2, 0.3],
                        [0.4, 0.5, 0.1],
                        [0.1, 0.1, 0.8]]]).astype('float64')


    tran_v = np.array([[0.4, 0.2, 0.4],
                       [0.3, 0.3, 0.4],
                       [0.5, 0.2, 0.3]]).astype('float64')

    ## start label to every label
    init_v = np.array([0.1, 0.5, 0.4]).astype('float64')

    gs_v = np.array([[1, 2, 1],
                     [1, 1, 2]]).astype('int64')

    masks_v = np.array([[1, 1, 1],
                        [1, 1, 1]]).astype('int64')

    # shape: (batch_size, time_steps, label_num)
    pred = theano.shared(name='pred', value=pred_v) # T.tensor3()
    init = theano.shared(name='init', value=init_v) # T.tensor3()
    # transition matrix, t-1 to t, with shape: (label_num + 1, label_num),
    # cause we also include the <start> label to other pre-defined label
    tran = theano.shared(name='tran', value=tran_v) # T.matrix() 
    # shape: (batch_size, time_steps)
    gs = theano.shared(name='gs', value=gs_v) # T.imatrix()
    masks = theano.shared(name='masks', value=masks_v) # T.imatrix()


    a = TagInferenceLayer(pred, init, tran, gs, masks)

    index_chain, last_step_max_score, last_gs_score = a.get_output_for() 





