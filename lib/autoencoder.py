import Data
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer

def init_params(config):
    params = {}

    scale = 0.1
    num_latent = 128

    params["W_enc_1"] = theano.shared(scale * np.random.normal(size = (4000, 2048)).astype('float32'))
    params["W_enc_2"] = theano.shared(scale * np.random.normal(size = (2048, 2048)).astype('float32'))

    params["b_enc_1"] = theano.shared(0.0 * np.random.normal(size = (2048)).astype('float32'))
    params["b_enc_2"] = theano.shared(0.0 * np.random.normal(size = (2048)).astype('float32'))

    params["z_mean_W"] = theano.shared(scale * np.random.normal(size = (2048, num_latent)).astype('float32'))
    params["z_mean_b"] = theano.shared(0.0 * np.random.normal(size = (num_latent)).astype('float32'))

    params["z_std_W"] = theano.shared(scale * np.random.normal(size = (2048, num_latent)).astype('float32'))
    params["z_std_b"] = theano.shared(0.0 * np.random.normal(size = (num_latent)).astype('float32'))

    params['W_dec_1'] = theano.shared(scale * np.random.normal(size = (num_latent, 2048)).astype('float32'))
    params['W_dec_2'] = theano.shared(scale * np.random.normal(size = (2048, 2048)).astype('float32'))
    params['W_dec_3'] = theano.shared(scale * np.random.normal(size = (2048, 4000)).astype('float32'))

    params['b_dec_1'] = theano.shared(0.0 * np.random.normal(size = (2048)).astype('float32'))
    params['b_dec_2'] = theano.shared(0.0 * np.random.normal(size = (2048)).astype('float32'))
    params['b_dec_3'] = theano.shared(0.0 * np.random.normal(size = (4000)).astype('float32'))

    return params

'''
Maps from a given x to an h_value.  


'''
#def encoder(x, params):
#    pass

'''
Maps from a given z to a decoded x.  

'''
#def decoder(z, params):
#    pass

'''
Given x (uunormalized), returns a reconstructed_x and a sampled x (both unnormalized)
'''

def define_network(x, params):

        h_out_1 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)

    results_map = {'reconstruction' : reconstruction}

def compute_loss(x, x_reconstructed):
    return T.mean(T.sqr((x - x_reconstructed)))

if __name__ == "__main__":

    params = init_params({})

    x = T.matrix()

    results_map = define_network(x, params)

    x_reconstructed = results_map['reconstruction']

    loss = compute_loss(x_reconstructed, x)

    inputs = [x]

    outputs = {'loss' : loss, 'reconstruction' : reconstruction}

    generator_gan_updates = lasagne.updates.adam(loss, params, learning_rate = 0.0001)

    train_method = theano.function(inputs = inputs, outputs = outputs, updates = updates)







