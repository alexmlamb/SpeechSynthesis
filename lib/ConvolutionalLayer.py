import numpy as np
import theano
import theano.tensor as T

from lasagne.theano_extensions.conv import conv1d_md, conv1d_sc, conv1d_mc0

import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# Set a fixed number for 2 purpose:
# Repeatable experiments; 2. for multiple-GPU, the same initial weights


class ConvPoolLayer(object):

    def __init__(self, in_channels, out_channels,
                 in_length, batch_size, kernel_len, W, b, stride = 1,
                 activation = "relu", batch_norm = False):

        self.stride = stride
        self.batch_norm = batch_norm
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_length = kernel_len
        self.in_length = in_length
        self.batch_size = batch_size

        self.filter_shape = np.asarray((out_channels, in_channels, kernel_len))

        self.W = W
        self.b = b

    def output(self, input):

        input = T.specify_shape(input, (self.batch_size, self.in_channels, self.in_length))

        conv_out = conv1d_mc0(input, self.W, image_shape = (self.batch_size, self.in_channels, self.in_length),
                                                filter_shape = (self.out_channels, self.in_channels, self.filter_length),
                                                subsample = (self.stride,))

        #was mb, filters, x, y
        #now mb, filters, x

        if self.batch_norm:
            conv_out = (conv_out - T.mean(conv_out, axis = (0,2), keepdims = True)) / (1.0 + T.std(conv_out, axis=(0,2), keepdims = True))

        conv_out += self.b.dimshuffle('x', 0, 'x')

        if self.activation == "relu":
            self.out = T.maximum(0.0, conv_out)
        elif self.activation == "tanh":
            self.out = T.tanh(conv_out)
        elif self.activation == None:
            self.out = conv_out

        return self.out




