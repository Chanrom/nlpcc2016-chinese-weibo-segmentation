import lasagne
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.layers import Layer
from lasagne.random import get_rng

class CWDropoutLayer(Layer):

    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(CWDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial((input_shape[0], input_shape[1]), p=retain_prob,
                                               dtype=input.dtype).dimshuffle(0, 1, 'x')
