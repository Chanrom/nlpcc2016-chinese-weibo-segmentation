#coding=utf-8
import theano
import numpy as np
import theano.tensor as T
from collections import OrderedDict


def get_or_compute_grads(loss_or_grads, params):

    if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
        raise ValueError("params must contain shared variables only. If it "
                         "contains arbitrary parameter expressions, then "
                         "lasagne.utils.collect_shared_vars() may help you.")
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)


# def l1_unit_norm(p):
#     epsilon = 10e-8
#     p = p * T.cast(p >= 0., 'float64')

#     return p / (epsilon + T.sum(p, axis=-1, keepdims=True))
    
    
def adagrad_norm(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6, constraints=None):

    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        #print param
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        
        new_param = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))

        if constraints.has_key(param):
            #print 'norm', param
            new_param = constraints[param](new_param)

        updates[param] = new_param

    return updates


def sgd_norm(loss_or_grads, params, learning_rate, constraints=None):
  
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        new_param = param - learning_rate * grad
        if constraints.has_key(param):
            new_param = constraints[param](new_param)
        updates[param] = new_param

    return updates