import lasagne
import theano.tensor as T
from lasagne.layers import Layer

class CWEmbeddingLayer(Layer):

    def __init__(self, incoming, input_size, output_size, window_size,
                 W=lasagne.init.Normal(), **kwargs):
        super(CWEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size # word dim
        self.window_size = window_size 

        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_size * self.window_size)

    def get_output_for(self, input, **kwargs):

        max_length = input.shape[1]

        out = self.W[input]

        return T.reshape(out, [-1, max_length, self.window_size * self.output_size])
