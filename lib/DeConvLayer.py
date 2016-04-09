
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np


def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


class DeConvLayer(object):

    def __init__(self, in_channels, out_channels, activation, up_rate, W, b, batch_norm = False):

        self.filter_shape = np.asarray((in_channels, out_channels, up_rate * 2 + 1, 1))

        self.activation = activation

        self.up_rate = up_rate

        self.batch_norm = batch_norm

        self.W = W
        self.b = b

    def output(self, input):

        conv_out = deconv(input, self.W, subsample=(self.up_rate, 1), border_mode=(self.up_rate,0))


        if self.batch_norm:
            conv_out = (conv_out - conv_out.mean(axis = (0,2,3), keepdims = True)) / (1.0 + conv_out.std(axis = (0,2,3), keepdims = True))

        conv_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        if self.activation == "relu":
            out = T.maximum(0.0, conv_out)
        elif self.activation == "tanh":
            out = T.tanh(conv_out)
        elif self.activation == None:
            out = conv_out
        else:
            raise Exception()


        return out



if __name__ == "__main__":

    x = T.tensor4()

    u = 20

    W = theano.shared(np.random.normal(size = (1, 10, u * 2 + 1, 1)).astype('float32'))
    b = theano.shared(np.random.normal(size = (10)).astype('float32'))

    dc = DeConvLayer(in_channels = 1, out_channels = 10, activation = None, W = W, b = b, up_rate = u)

    x_gen = np.ones(shape = (32, 1, 200, 1)).astype('float32')

    print "compiling"

    f = theano.function([x], dc.output(x))

    val = f(x_gen)
    print val.shape



