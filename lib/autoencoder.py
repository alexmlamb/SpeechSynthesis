from Data import Data
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer

def init_params(config):
    params = {}

    scale = 0.1
    num_latent = 128

    num_hidden = config['num_hidden']

    params["W_enc_1"] = theano.shared(scale * np.random.normal(size = (4000, num_hidden)).astype('float32'))
    params["W_enc_2"] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))

    params["b_enc_1"] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params["b_enc_2"] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))

    #params["z_mean_W"] = theano.shared(scale * np.random.normal(size = (2048, num_latent)).astype('float32'))
    #params["z_mean_b"] = theano.shared(0.0 * np.random.normal(size = (num_latent)).astype('float32'))

    #params["z_std_W"] = theano.shared(scale * np.random.normal(size = (2048, num_latent)).astype('float32'))
    #params["z_std_b"] = theano.shared(0.0 * np.random.normal(size = (num_latent)).astype('float32'))

    #params['W_dec_1'] = theano.shared(scale * np.random.normal(size = (num_latent, 2048)).astype('float32'))
    params['W_dec_2'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_dec_3'] = theano.shared(scale * np.random.normal(size = (num_hidden, 4000)).astype('float32'))

    #params['b_dec_1'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_dec_2'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_dec_3'] = theano.shared(0.0 * np.random.normal(size = (4000)).astype('float32'))

    return params

def init_params_disc(config):

    params = {}

    scale = 0.05
    num_hidden = config['num_hidden']

    params['W_disc_1'] = theano.shared(scale * np.random.normal(size = (4000, num_hidden)).astype('float32'))
    params['W_disc_2'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_disc_3'] = theano.shared(scale * np.random.normal(size = (num_hidden, 1)).astype('float32'))

    params['b_disc_1'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_disc_2'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_disc_3'] = theano.shared(0.0 * np.random.normal(size = (1)).astype('float32'))

    print "DEFINED ALL DISCRIMINATOR WEIGHTS"

    return params

def normalize(x):
    return x / 10000.0

def denormalize(x):
    return x * 10000.0

'''
Takes real samples and generated samples.  

Two fully connected layers, then a classifier output.  


'''
def discriminator(x_real, x_generated, params, mb_size, num_hidden):

    x = T.concatenate([x_real, x_generated], axis = 0)

    target = theano.shared(np.asarray([1] * mb_size + [0] * mb_size).astype('int32'))

    h_out_1 = DenseLayer((mb_size * 2, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify, W = params['W_disc_1'], b = params['b_disc_1'])

    h_out_2 = DenseLayer((mb_size * 2, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify, W = params['W_disc_2'], b = params['b_disc_2'])

    h_out_3 = DenseLayer((mb_size * 2, num_hidden), num_units = num_hidden, nonlinearity=None, W = params['W_disc_3'], b = params['b_disc_3'])

    h_out_1_value = h_out_1.get_output_for(x)
    h_out_2_value = h_out_2.get_output_for(h_out_1_value)
    h_out_3_value = h_out_3.get_output_for(h_out_2_value)

    raw_y = T.clip(h_out_3_value, -20.0, 20.0)

    classification = T.nnet.sigmoid(raw_y)

    loss = -1.0 * (target * -1.0 * T.log(1 + T.exp(-1.0*raw_y.flatten())) + (1 - target) * (-raw_y.flatten() - T.log(1 + T.exp(-raw_y.flatten()))))

    accuracy = T.mean(T.eq(target, T.gt(classification, 0.5).flatten()))

    results = {'loss' : T.mean(loss), 'accuracy' : accuracy, 'c' : classification}

    return results

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

def define_network(x, params, config):

    num_hidden = config['num_hidden']
    mb_size = config['mb_size']


    h_out_1 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify, W = params['W_enc_1'], b = params['b_enc_1'])
    h_out_2 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify, W = params['W_enc_2'], b = params['b_enc_2'])
    h_out_3 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify, W = params['W_dec_2'], b = params['b_dec_2'])
    h_out_4 = DenseLayer((mb_size, 4000), num_units = num_hidden, nonlinearity=None, W = params['W_dec_3'], b = params['b_dec_3'])

    h_out_1_value = h_out_1.get_output_for(x)
    h_out_2_value = h_out_2.get_output_for(h_out_1_value)
    h_out_3_value = h_out_3.get_output_for(h_out_2_value)
    reconstruction = h_out_4.get_output_for(h_out_3_value)

    results_map = {'reconstruction' : reconstruction}

    return results_map

def compute_loss(x, x_reconstructed):
    return T.mean(T.sqr((x - x_reconstructed)))# + T.abs_(T.mean(x) - T.mean(x_reconstructed)) + T.mean(T.abs_(T.std(x) - T.std(x_reconstructed)))

if __name__ == "__main__":

    config = {}
    config['mb_size'] = 128
    config['num_hidden'] = 4096

    d = Data(mb_size = config['mb_size'], seq_length = 4000)

    #todo make sure disc is only updating on disc_params.  

    params = init_params(config)
    params_disc = init_params_disc(config)

    x = T.matrix()

    results_map = define_network(normalize(x), params, config)

    x_reconstructed = results_map['reconstruction']

    disc_results = discriminator(normalize(x), normalize(x_reconstructed), params_disc, mb_size = config['mb_size'], num_hidden = config['num_hidden'])

    loss = compute_loss(normalize(x_reconstructed), normalize(x))

    inputs = [x]

    outputs = {'loss' : loss, 'reconstruction' : x_reconstructed, 'accuracy' : disc_results['accuracy'], 'classification' : disc_results['c']}

    print "params_gen", params.keys()
    print "params_disc", params_disc.keys()

    updates = lasagne.updates.adam(loss, params.values(), learning_rate = 0.0)
    updates_disc = lasagne.updates.adam(disc_results['loss'], params_disc.values(), learning_rate = 0.0001, beta1 = 0.5)
    updates_gen = lasagne.updates.adam(-1.0 * disc_results['loss'], params.values(), learning_rate = 0.001, beta1 = 0.5)

    train_method = theano.function(inputs = inputs, outputs = outputs, updates = updates)
    disc_method = theano.function(inputs = inputs, outputs = outputs, updates = updates_disc)
    gen_method = theano.function(inputs = inputs, outputs = outputs, updates = updates_gen)

    last_acc = 0.0

    for i in range(0,100000):
        x = d.getBatch()
        res = train_method(x)
        if last_acc < 0.8:
            disc_method(x)
        if last_acc > 0.5:
            gen_method(x)

        last_acc = res['accuracy']

        if i % 100 == 1:
            d.saveExample(res['reconstruction'][0][:200], "image_reconstruction")
            d.saveExample(x[0][:200], "image_original")

            recon = res['reconstruction'][0]
            true = x[0]

            d.saveExampleWav(res['reconstruction'][0], "image_reconstruction")
            d.saveExampleWav(x[0], "image_original")

            print "==========================================================="
            print ""

            print "update", i, "loss", res['loss']

            print "accuracy d", res['accuracy']





