import numpy as np
import theano.tensor as T
import theano


class TagInferenceLayer(object):
    """
    Added by Chanrom, 12.20.2015
    lasagne.layers.TagInferenceLayer(incoming, num_labels, tran_t=lasagne.init.GlorotUnifrom,
    init_t=lasagne.init.Constant(0.), gs_t=lasagne.init.Constant(0.), mask_input=None, **kwargs)

    A tag inference layer

    Parameters
    ----------
    incoming : a :class:'Layer' instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_labels : int
        The number of labels 

    tran_t : Theano shared variable, expression, numpy array or callable or 'None'
        Initial value, expression or initializer for the transition matrix for tags.
        This should be a matrix with shape '(num_label, num_label)'.

    init_t : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the initial transition score.

    gs_t : Theano shared variable, expression, numpy array, callable or 'None'
        Golden standard label sequence for caculating score of gs sequence.
        This should be a matrix with shape '(batch size, max sequence length)'

    mask_input : Theano shared variable, expression, numpy array, callable or 'None'
        mask matrix, this should be a matrix with shape '(batch size, max sequence length)'
    """
    def __init__(self, margin_discount, pred_t, init_t=None, tran_t=None,
                 gs_t=None, mask_input=None, **kwargs):
        self.margin_discount = margin_discount
        self.pred = pred_t
        self.tran_t = tran_t
        self.init_t = init_t
        self.gs = gs_t
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


    def score_one_step(self, pred_t, gs_t, mask_t, score_tm1, index_tm1, init_tran, tran, deterministic, init_flag):
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


        def scan_example(pred_t_e, gs_t_e, mask_t_e, score_tm1_e, index_tm1_e, init_tran, tran, deterministic, init_flag):
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


            ## !!! THIS TAG INFERENCE LAYER IS ONLY SUITABLE FOR CWS !!!
            ## OUR LABELS ARE (0, B), (1, E), (2, S), (3, I)
            def scan_label(index, gs_t_e, pred_t_e, score_tm1_e, init_tran, tran, deterministic, init_flag):
                # index: (1, )
                # gs_t_e: (1, )
                # pred_t_e: (label_num, )
                # score_tm1_e: (label_num, )

                from_index = None
                score_t_e_l = None
                delta_loss = None #T.constant(0)
                print 'scan label'
                delta_loss = theano.ifelse.ifelse(T.neq(gs_t_e, index),
                                      T.constant(self.margin_discount),
                                      T.constant(0.0))
                # if T.neq(gs_t_e, index) == T.constant(1):
                #     delta_loss = T.constant(self.margin_discount)
                # print 'delta_loss', delta_loss.eval()
                print 'init_flag', init_flag.eval()
                if init_flag == T.constant(1): # initial step, <start> -> B, S are valid
                    print 'init step'

                    from_index = T.constant(-1)

                    score_t_e_l = theano.ifelse.ifelse(T.eq(index, 0),
                        score_tm1_e[0] + init_tran[0] + pred_t_e[0] + (1 - deterministic) * delta_loss,
                        score_tm1_e[0] + init_tran[2] + pred_t_e[2] + (1 - deterministic) * delta_loss)


                    # if T.eq(index, 0) == T.constant(1): # current label is B
                    #     # only one path which is comming from <start> 
                    #     from_index = T.constant(-1) # <start> index doesn't matter
                    #     score_t_e_l = T.constant(score_tm1_e[0]) + init_tran[0] + pred_t_e[0] + (1 - deterministic) * delta_loss
                    # elif T.eq(index, 2) == T.constant(1): # current label is S
                    #     from_index = T.constant(-1)
                    #     score_t_e_l = T.constant(score_tm1_e[0]) + init_tran[2] + pred_t_e[2] + (1 - deterministic) * delta_loss

                elif init_flag == T.constant(0):
                    print 'other step'
                    #if T.eq(index, 0) == T.constant(0): # current label is B
                    print 'index == T.constant(0)', index == T.constant(0)


                    # from_index, score_t_e_l = \
                    #     theano.ifelse.ifelse(T.eq(index, T.constant(0)),
                    #         theano.ifelse.ifelse(T.ge(score_tm1_e[2] + tran[2][0] + pred_t_e[0] + (1 - deterministic) * delta_loss, score_tm1_e[1] + tran[1][0] + pred_t_e[0] + (1 - deterministic) * delta_loss),
                    #             [T.constant(2), score_tm1_e[2] + tran[2][0] + pred_t_e[0] + (1 - deterministic) * delta_loss],
                    #             [T.constant(1), score_tm1_e[1] + tran[1][0] + pred_t_e[0] + (1 - deterministic) * delta_loss]),
                    #         theano.ifelse.ifelse(T.eq(index, T.constant(1)),
                    #             theano.ifelse.ifelse(T.ge(score_tm1_e[0] + tran[0][1] + pred_t_e[1] + (1 - deterministic) * delta_loss, score_tm1_e[3] + tran[3][1] + pred_t_e[1] + (1 - deterministic) * delta_loss),
                    #                 [T.constant(0), score_tm1_e[0] + tran[0][1] + pred_t_e[1] + (1 - deterministic) * delta_loss],
                    #                 [T.constant(3), score_tm1_e[3] + tran[3][1] + pred_t_e[1] + (1 - deterministic) * delta_loss]),
                    #             theano.ifelse.ifelse(T.eq(index, T.constant(2)),
                    #                 theano.ifelse.ifelse(T.ge(score_tm1_e[2] + tran[2][2] + pred_t_e[2] + (1 - deterministic) * delta_loss, score_tm1_e[1] + tran[1][2] + pred_t_e[2] + (1 - deterministic) * delta_loss),
                    #                     [T.constant(2), score_tm1_e[2] + tran[2][2] + pred_t_e[2] + (1 - deterministic) * delta_loss],
                    #                     [T.constant(1), score_tm1_e[1] + tran[1][2] + pred_t_e[2] + (1 - deterministic) * delta_loss]),
                    #                 theano.ifelse.ifelse(T.eq(index, T.constant(3)),
                    #                     theano.ifelse.ifelse(T.ge(score_tm1_e[0] + tran[0][3] + pred_t_e[3] + (1 - deterministic) * delta_loss, score_tm1_e[3] + tran[3][3] + pred_t_e[3] + (1 - deterministic) * delta_loss),
                    #                         [T.constant(0), score_tm1_e[0] + tran[0][3] + pred_t_e[3] + (1 - deterministic) * delta_loss],
                    #                         [T.constant(3), score_tm1_e[3] + tran[3][3] + pred_t_e[3] + (1 - deterministic) * delta_loss]), [T.constant(0), T.constant(0)]
                    #                     )
                    #                 )
                    #             )
                    #         )


                    from_index, score_t_e_l = \
                        theano.ifelse.ifelse(T.eq(index, T.constant(0)),
                            theano.ifelse.ifelse(T.ge(score_tm1_e[2] + tran[2][0] + pred_t_e[0] + (1 - deterministic) * delta_loss, score_tm1_e[1] + tran[1][0] + pred_t_e[0] + (1 - deterministic) * delta_loss),
                                [T.constant(2), score_tm1_e[2] + tran[2][0] + pred_t_e[0] + (1 - deterministic) * delta_loss],
                                [T.constant(1), score_tm1_e[1] + tran[1][0] + pred_t_e[0] + (1 - deterministic) * delta_loss]), [T.constant(0), T.constant(0)])

                    if index == T.constant(0):
                        # two paths: S -> B or E -> B
                        s_score = score_tm1_e[2] + tran[2][0] + pred_t_e[0] + (1 - deterministic) * delta_loss
                        e_score = score_tm1_e[1] + tran[1][0] + pred_t_e[0] + (1 - deterministic) * delta_loss
                        if T.ge(s_score, e_score) == T.constant(1): # s_score > e_score
                            score_t_e_l = s_score
                            from_index = T.constant(2)
                        else:
                            score_t_e_l = e_score
                            from_index = T.constant(1)
                    elif T.eq(index, 1) == T.constant(1): # current label is E
                        # two paths: B -> E or I -> E
                        b_score = score_tm1_e[0] + tran[0][1] + pred_t_e[1] + (1 - deterministic) * delta_loss
                        i_score = score_tm1_e[3] + tran[3][1] + pred_t_e[1] + (1 - deterministic) * delta_loss
                        if T.ge(b_score, i_score) == T.constant(1):
                            score_t_e_l = b_score
                            from_index = T.constant(0)
                        else:
                            score_t_e_l = i_score
                            from_index = T.constant(3)
                    elif T.eq(index, 2) == T.constant(1): # current label is S
                        # two paths: S -> S or E -> S
                        s_score = score_tm1_e[2] + tran[2][2] + pred_t_e[2] + (1 - deterministic) * delta_loss
                        e_score = score_tm1_e[1] + tran[1][2] + pred_t_e[2] + (1 - deterministic) * delta_loss
                        if T.ge(s_score, e_score) == T.constant(1):
                            score_t_e_l = s_score
                            from_index = T.constant(2)
                        else:
                            score_t_e_l = e_score
                            from_index = T.constant(1)
                    elif T.eq(index, 3) == T.constant(1): # current label is I
                        # two paths: B -> I or I -> I
                        b_score = score_tm1_e[0] + tran[0][3] + pred_t_e[3] + (1 - deterministic) * delta_loss
                        i_score = score_tm1_e[3] + tran[3][3] + pred_t_e[3] + (1 - deterministic) * delta_loss
                        if T.ge(b_score, i_score) == T.constant(1):
                            score_t_e_l = b_score
                            from_index = T.constant(0)
                        else:
                            score_t_e_l = i_score
                            from_index = T.constant(3)

                print 'return type scan label', type(from_index), type(score_t_e_l)
                return from_index, score_t_e_l


            label_num = pred_t_e.shape[0]

            score_t_e = None
            index_t_e = None
            print 'init_flag',  init_flag.eval()

            # score_t0_e = [T.constant(0), T.constant(-np.inf), T.constant(-np.inf), T.constant(-np.inf)]
            # (x, y) = T.switch(#index_t_e, score_t_e
            #     T.eq(init_flag, T.constant(1)),
            #     theano.scan(fn=scan_label,
            #                 sequences=[T.arange(label_num)],
            #                 non_sequences=[gs_t_e, pred_t_e, score_t0_e,
            #                 init_tran, tran, deterministic, init_flag])[0], 
            #     theano.scan(fn=scan_label,
            #                 sequences=[T.arange(label_num)],
            #                 non_sequences=[gs_t_e, pred_t_e, score_tm1_e,
            #                 init_tran, tran, deterministic, init_flag])[0]
            #     )

            # print 'x', type(x[0])

            if init_flag == T.constant(1):#T.eq(init_flag, T.constant(1)): # initial step
                print 'scan_example init_step'
                # we only have one label <start>, other paths are invalid
                score_t0_e = [0.0, T.constant(-np.inf), T.constant(-np.inf), T.constant(-np.inf)]
                # return shape: (label num, )
                index_t_e, score_t_e = theano.scan(fn=scan_label,
                                        sequences=[T.arange(label_num)],
                                        non_sequences=[gs_t_e, pred_t_e, score_t0_e, init_tran, tran, deterministic, init_flag])[0]
            elif init_flag == T.constant(0):#T.eq(init_flag, T.constant(0)): # other            
                print 'scan_example other'
                index_t_e, score_t_e = theano.scan(fn=scan_label,
                                        sequences=[T.arange(label_num)],
                                        non_sequences=[gs_t_e, pred_t_e, score_tm1_e, init_tran, tran, deterministic, init_flag])[0]

            score_t_e = score_t_e * mask_t_e + score_tm1_e * (1 - mask_t_e)

            index_t_e = T.cast(index_t_e * mask_t_e + T.arange(label_num) * (1 - mask_t_e), 'int64')

            return score_t_e, index_t_e


        print 'init_flag', init_flag.eval()

        # return shape: (batch size, label num)...
        score_t, index_t = theano.scan(fn=scan_example,
                                        sequences=[pred_t, gs_t, mask_t, score_tm1, index_tm1],
                                        non_sequences=[init_tran, tran, deterministic, init_flag])[0]

        return score_t, index_t       


    def get_output_for(self, deterministic=False):

        if deterministic:
            deterministic_flag = T.constant(1)
        else:
            deterministic_flag = T.constant(0)

        batch_size = self.pred.shape[0]
        time_steps = self.pred.shape[1]
        label_num = self.pred.shape[2]

        ## the start state to first label
        pred_t1 = self.pred[:, 0] # shape: (batch size, label num)
        gs_t1 = self.gs[:, 0] - 1
        mask_t1 = self.masks[:, 0]

        score_t0 = T.zeros((batch_size, label_num))
        index_t0 = T.zeros((batch_size, label_num), dtype='int64')

        init_flag = T.constant(1)
        # return shape: (batch size, label num), (batch size, label num)
        score_t1, index_t1 = self.score_one_step(pred_t1, gs_t1,
            mask_t1, score_t0, index_t0, self.init_t, self.tran_t, deterministic_flag, init_flag)

        print 'score_t1', score_t1.eval()
        print 'index_t1', index_t1.eval()

        pred = self.pred.dimshuffle(1, 0, 2)
        gs = self.gs.dimshuffle(1, 0)
        mask = self.masks.dimshuffle(1, 0)
        init_flag = T.constant(0)

        print pred[1:].eval().shape
        print (gs[1:]-1).eval().shape
        print mask[1:].eval().shape
        # return shape: (time steps - 1, batch size, label num) ..., (time steps - 1, batch size)
        step_scores, step_indexs = theano.scan(fn=self.score_one_step,
                                               outputs_info=[score_t1, index_t1],
                                               sequences=[pred[1:], gs[1:]-1, mask[1:]],
                                               non_sequences=[self.init_t, self.tran_t, deterministic_flag, init_flag])[0]

        print step_scores.eval().shape
        print step_indexs.eval().shape
        print 'score_t2', step_scores.dimshuffle(1, 0, 2)[:, 0].eval()
        print 'index_t2', step_indexs.dimshuffle(1, 0, 2)[:, 0].eval()
        print 'score_t3', step_scores.dimshuffle(1, 0, 2)[:, 1].eval()
        print 'index_t3', step_indexs.dimshuffle(1, 0, 2)[:, 1].eval()

        # shape: (batch size, )
        last_step_max_score = T.max(step_scores[-1], axis=-1)
        last_step_max_index = T.argmax(step_scores[-1], axis=-1)

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

        # reverse time step, shape: (time steps - 1, batch size, label num)
        #step_indexs = step_indexs[::-1]

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

                cost_t_e = None
                if T.eq(step_index, 0) == T.constant(1):
                    cost_t_e = pred_t_e[index_chain_t_e] + init_tran[index_chain_t_e]\
                     - pred_t_e[gs_t_e] - init_tran[gs_t_e]
                else:
                    cost_t_e = pred_t_e[index_chain_t_e] + tran[index_chain_t_e][index_chain_tm1_e]\
                     - pred_t_e[gs_t_e] - tran[gs_tm1_e][gs_t_e]

                cost_t_e = cost_t_e * mask_t_e + cost_tm1_e * (1 - mask_t_e)

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
        gs_t0 = T.zeros((batch_size, ), dtype='int64')
        cost_t0 = T.zeros((batch_size, ), dtype='float64')
        index_chain_t0 = T.zeros((batch_size, ), dtype='int64')

        # return shape: (time steps, batch size)
        steps_cost, _, _ = theano.scan(fn=one_step_cost,
                                 outputs_info=[cost_t0, gs_t0, index_chain_t0],
                                 sequences=[T.arange(time_steps), pred, gs-1, index_chain_sff-1, mask],
                                 non_sequences=[self.init_t, self.tran_t])[0]

        # return shape: (batch size, )
        cost = T.sum(steps_cost.dimshuffle(1, 0), axis=-1)

        # # return shape: (batch size, time steps - 1)                                                                                                                                      
        # step_gs_scores = step_gs_scores.dimshuffle(1, 0)

        # # return shape: (batch size, )                                                                                                                                                    
        # last_gs_score = step_gs_scores[:, -1]

        # print 'score_t2', step_scores.dimshuffle(1, 0, 2)[:, 0].eval()
        # print 'index_t2', step_indexs.dimshuffle(1, 0, 2)[:, 0].eval()
        # print 'gs_score_t2', step_gs_scores[:, 0].eval()

        # print 'score_t3', step_scores.dimshuffle(1, 0, 2)[:, 1].eval()
        # print 'index_t3', step_indexs.dimshuffle(1, 0, 2)[:, 1].eval()
        # print 'gs_score_t3', step_gs_scores[:, 1].eval()

        # print index_chain.eval()
        # print last_step_max_score.eval()
        # print last_gs_score.eval()        

        # return shape: (exmaple num, time steps), (batch size, ), (batch size, )
        #return [index_chain, last_step_max_score, last_gs_score]

        print 'cost', cost.eval()
        # return shape: (batch size, )
        return cost


if __name__=='__main__':

    pred_v = np.array([[[0.2, 0.6, 0.1, 0.1],
                        [0.1, 0.3, 0.5, 0.1],
                        [0.1, 0.1, 0.1, 0.7]],
                       [[0.5, 0.2, 0.2, 0.1],
                        [0.4, 0.4, 0.1, 0.1],
                        [0.1, 0.1, 0.6, 0.2]]]).astype('float64')


    tran_v = np.array([[-100, 0.5, -100, 0.5],
                       [0.5, -100, 0.5, -100],
                       [0.5, -100, 0.5, -100],
                       [-100, 0.5, -100, 0.5]]).astype('float64')

    ## start label to every label
    init_v = np.array([0.5, -100, 0.5, -100]).astype('float64')

    gs_v = np.array([[1, 2, 3],
                     [3, 3, 3]]).astype('int64')

    masks_v = np.array([[1, 1, 1],
                        [1, 1, 1]]).astype('int64')

    # shape: (batch_size, time_steps, label_num)
    pred_s = theano.shared(name='pred', value=pred_v) # T.tensor3()
    init_s = theano.shared(name='init', value=init_v) # T.tensor3()
    # transition matrix, t-1 to t, with shape: (label_num + 1, label_num),
    # cause we also include the <start> label to other pre-defined label
    tran_s = theano.shared(name='tran', value=tran_v) # T.matrix() 
    # shape: (batch_size, time_steps)
    gs_s = theano.shared(name='gs', value=gs_v) # T.imatrix()
    masks_s = theano.shared(name='masks', value=masks_v) # T.imatrix()


    a = TagInferenceLayer(np.float32(1), pred_s, init_s, tran_s, gs_s, masks_s)

    cost = a.get_output_for()
