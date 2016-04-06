from Data import Data
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer
from HiddenLayer import HiddenLayer
from ConvolutionalLayer import ConvPoolLayer
import time

from consider_constant import consider_constant

def init_params_encoder(config):
    params = {}

    scale = 0.01
    num_latent = config['num_latent']

    num_hidden = config['num_hidden']

    params["W_enc_1"] = theano.shared(scale * np.random.normal(size = (4000, num_hidden)).astype('float32'))
    params["W_enc_2"] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))

    params["b_enc_1"] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params["b_enc_2"] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))

    params["z_mean_W"] = theano.shared(scale * np.random.normal(size = (num_hidden, num_latent)).astype('float32'))
    params["z_mean_b"] = theano.shared(0.0 * np.random.normal(size = (num_latent)).astype('float32'))

    #params["z_std_W"] = theano.shared(scale * np.random.normal(size = (num_hidden, num_latent)).astype('float32'))
    #params["z_std_b"] = theano.shared(0.0 * np.random.normal(size = (num_latent)).astype('float32'))

    return params

def init_params_decoder(config):

    params = {}

    scale = 0.01
    num_latent = config['num_latent']
    num_hidden = config['num_hidden']

    params['W_dec_1'] = theano.shared(scale * np.random.normal(size = (num_latent * 2, num_hidden)).astype('float32'))
    params['W_dec_2'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_dec_3'] = theano.shared(scale * np.random.normal(size = (num_hidden, 4000)).astype('float32'))

    params['b_dec_1'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_dec_2'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_dec_3'] = theano.shared(0.0 * np.random.normal(size = (4000)).astype('float32'))

    return params

def init_params_disc(config):

    params = {}

    scale = 0.01
    num_hidden = config['num_hidden']
    num_latent = config['num_latent']

    params["W_c_1"] = theano.shared(scale * np.random.normal(size = (128, 1, 20)).astype('float32'))
    params["b_c_1"] = theano.shared(scale * np.random.normal(size = (128)).astype('float32'))

    params["W_c_2"] = theano.shared(scale * np.random.normal(size = (256, 128, 20)).astype('float32'))
    params["b_c_2"] = theano.shared(scale * np.random.normal(size = (256)).astype('float32'))

    params["W_c_3"] = theano.shared(scale * np.random.normal(size = (512, 256, 10)).astype('float32'))
    params["b_c_3"] = theano.shared(scale * np.random.normal(size = (512)).astype('float32'))

    params["W_ch_1"] = theano.shared(scale * np.random.normal(size = (512 * 6, num_hidden)).astype('float32'))
    params["b_ch_1"] = theano.shared(scale * np.random.normal(size = (num_hidden)).astype('float32'))

    params['W_disc_1'] = theano.shared(scale * np.random.normal(size = (num_hidden + num_latent, num_hidden)).astype('float32'))
    params['W_disc_2'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_disc_3'] = theano.shared(scale * np.random.normal(size = (num_hidden, num_hidden)).astype('float32'))
    params['W_disc_4'] = theano.shared(scale * np.random.normal(size = (num_hidden, 1)).astype('float32'))

    params['b_disc_1'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_disc_2'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_disc_3'] = theano.shared(0.0 * np.random.normal(size = (num_hidden)).astype('float32'))
    params['b_disc_4'] = theano.shared(0.0 * np.random.normal(size = (1)).astype('float32'))

    print "DEFINED ALL DISCRIMINATOR WEIGHTS"

    return params

def normalize(x):
    return x / 1000.0

def denormalize(x):
    return x * 1000.0

'''
Takes real samples and generated samples.  

Two fully connected layers, then a classifier output.  

Takes x_real, x_reconstructed, and z (reconstruction)

Feed both (x_real, z) and (x_reconstructed, z) to the discriminator.  

Run three times: D_real, D_fake, G_fake.  When we run with G_fake, pass in consider_constant(z).  

'''
def discriminator(x, z, params, mb_size, num_hidden, num_latent):

    c_1 = ConvPoolLayer(in_channels = 1, out_channels = 128, in_length = 4000, batch_size = mb_size, kernel_len = 20, stride = 10, activation = "relu", batch_norm = True, W = params['W_c_1'], b = params['b_c_1'])

    c_2 = ConvPoolLayer(in_channels = 128, out_channels = 256, in_length = 399, batch_size = mb_size, kernel_len = 20, stride = 10, activation = "relu", batch_norm = True, W = params['W_c_2'], b = params['b_c_2'])

    c_3 = ConvPoolLayer(in_channels = 256, out_channels = 512, in_length = 38, batch_size = mb_size, kernel_len = 10, stride = 5, activation = "relu", batch_norm = True, W = params['W_c_3'], b = params['b_c_3'])

    c_h_1 = HiddenLayer(num_in = 6 * 512, num_out = num_hidden, W = params['W_ch_1'], b = params['b_ch_1'], activation = 'relu', batch_norm = True)

    h_out_1 = DenseLayer((mb_size, num_hidden + num_latent), num_units = num_hidden, nonlinearity=None, W = params['W_disc_1'])

    h_out_2 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=None, W = params['W_disc_2'])

    h_out_3 = DenseLayer((mb_size, num_hidden), num_units = num_hidden, nonlinearity=None, W = params['W_disc_3'])

    h_out_4 = DenseLayer((mb_size, 1), num_units = 1, nonlinearity=None, W = params['W_disc_4'], b = params['b_disc_4'])

    c_1_value = T.specify_shape(c_1.output(x.reshape((128,1,4000))), (128,128,399))

    c_2_value = T.specify_shape(c_2.output(c_1_value), (128,256,38))

    c_3_value = T.specify_shape(c_3.output(c_2_value), (128,512,6))

    c_h_1_value = c_h_1.output(c_3_value.flatten(2))

    h_out_1_value = h_out_1.get_output_for(0.0 * T.sum(c_1_value) + T.concatenate([z, c_h_1_value], axis = 1))

    h_out_1_value = T.maximum(0.0, (h_out_1_value - T.mean(h_out_1_value, axis = 0)) / (1.0 + T.std(h_out_1_value, axis = 0)) + params['b_disc_1'])

    h_out_2_value = h_out_2.get_output_for(h_out_1_value)

    h_out_2_value = T.maximum(0.0, (h_out_2_value - T.mean(h_out_2_value, axis = 0)) / (1.0 + T.std(h_out_2_value, axis = 0)) + params['b_disc_2'])

    h_out_3_value = h_out_3.get_output_for(h_out_2_value)

    h_out_3_value = T.maximum(0.0, (h_out_3_value - T.mean(h_out_3_value, axis = 0)) / (1.0 + T.std(h_out_3_value, axis = 0)) + params['b_disc_3'])

    h_out_4_value = h_out_4.get_output_for(h_out_3_value)

    raw_y = h_out_4_value

    classification = T.nnet.sigmoid(raw_y)

    results = {'c' : classification}

    return results

'''
Maps from a given x to an h_value.  


'''
def encoder(x, params, config):

    mb_size = config['mb_size']
    num_hidden = config['num_hidden']

    h_out_1 = HiddenLayer(num_in = 4000, num_out = num_hidden, W = params['W_enc_1'], b = params['b_enc_1'], activation = 'relu', batch_norm = True)

    h_out_2 = HiddenLayer(num_in = num_hidden, num_out = num_hidden, W = params['W_enc_2'], b = params['b_enc_2'], activation = 'relu', batch_norm = True)

    h_out_1_value = h_out_1.output(x)
    h_out_2_value = h_out_2.output(h_out_1_value)

    return {'h' : h_out_2_value}

'''
Maps from a given z to a decoded x.  

'''
def decoder(z, z_extra, params, config):

    mb_size = config['mb_size']
    num_latent = config['num_latent']
    num_hidden = config['num_hidden']

    h_out_1 = HiddenLayer(num_in = num_latent, num_out = num_hidden, W = params['W_dec_1'], b = params['b_dec_1'], activation = 'relu', batch_norm = True)

    h_out_2 = HiddenLayer(num_in = num_hidden, num_out = num_hidden, W = params['W_dec_2'], b = params['b_dec_2'], activation = 'relu', batch_norm = True)

    h_out_3 = DenseLayer((mb_size, num_hidden), num_units = 4000, nonlinearity=None, W = params['W_dec_3'], b = params['b_dec_3'])

    z = T.concatenate([z,z_extra], axis = 1)

    h_out_1_value = h_out_1.output(z)
    h_out_2_value = h_out_2.output(h_out_1_value)
    h_out_3_value = h_out_3.get_output_for(h_out_2_value)

    return {'h' : h_out_3_value}

'''
Given x (unormalized), returns a reconstructed_x and a sampled x (both unnormalized)
'''

def define_network(x, params, config):

    num_hidden = config['num_hidden']
    mb_size = config['mb_size']
    num_latent = config['num_latent']

    enc = encoder(x, params, config)

    mean_layer = DenseLayer((mb_size, num_hidden), num_units = num_latent, nonlinearity=None, W = params['z_mean_W'], b = params['z_mean_b'])
    #std_layer = DenseLayer((mb_size, num_hidden), num_units = num_latent, nonlinearity=None, W = params['z_std_W'], b = params['z_std_b'])

    mean = mean_layer.get_output_for(enc['h'])
    #std = T.exp(std_layer.get_output_for(enc['h']))

    import random as rng
    srng = theano.tensor.shared_randomstreams.RandomStreams(420)

    z_sampled = srng.normal(size = mean.shape, avg = 0.0, std = 1.0)
    z_extra = srng.normal(size = mean.shape, avg = 0.0, std = 1.0)

    z_reconstruction = mean

    #z_var = std**2
    z_loss = 0.0 * T.sum(mean)#0.001 * 0.5 * T.sum(mean**2 + z_var - T.log(z_var) - 1.0)

    dec_reconstruction = decoder(z_reconstruction, z_extra, params, config)
    dec_sampled = decoder(z_sampled, z_extra, params, config)

    interp_lst = []

    for j in range(0,128):
        interp_lst.append(z_reconstruction[0] * (j/128.0) + z_reconstruction[-1] * (1 - j / 128.0))

    z_interp = T.concatenate([interp_lst], axis = 1)

    dec_interp = decoder(z_interp, z_extra, params, config)

    results_map = {'reconstruction' : dec_reconstruction['h'], 'z_loss' : z_loss, 'sample' : dec_sampled['h'], 'interp' : dec_interp['h'], 'z' : z_reconstruction}

    return results_map

def compute_loss(x, x_reconstructed):
    return 100.0 * T.mean(T.sqr((x - x_reconstructed)))# + T.abs_(T.mean(x) - T.mean(x_reconstructed)) + T.mean(T.abs_(T.std(x) - T.std(x_reconstructed)))

if __name__ == "__main__":

    config = {}
    config['mb_size'] = 128
    config['num_hidden'] = 4096
    config['num_latent'] = 4096

    d = Data(mb_size = config['mb_size'], seq_length = 4000)

    #todo make sure disc is only updating on disc_params.  

    params_enc = init_params_encoder(config)
    params_dec = init_params_decoder(config)
    params_disc = init_params_disc(config)

    params = {}
    params.update(params_enc)
    params.update(params_dec)

    for pv in params_enc.values() + params_disc.values() + params_dec.values():
        print pv.dtype

    x = T.matrix()

    results_map = define_network(normalize(x), params, config)

    x_reconstructed = results_map['reconstruction']
    x_sampled = results_map['sample']

    disc_real_D = discriminator(normalize(x), results_map['z'], params_disc, mb_size = config['mb_size'], num_hidden = config['num_hidden'], num_latent = config['num_latent'])
    disc_fake_D = discriminator(x_reconstructed, results_map['z'], params_disc, mb_size = config['mb_size'], num_hidden = config['num_hidden'], num_latent = config['num_latent'])
    disc_fake_G = discriminator(x_reconstructed, consider_constant(results_map['z']), params_disc, mb_size = config['mb_size'], num_hidden = config['num_hidden'], num_latent = config['num_latent'])

    bce = T.nnet.binary_crossentropy

    LD_dD = bce(disc_real_D['c'], 0.99 * T.ones(disc_real_D['c'].shape)).mean() + bce(disc_fake_D['c'], 0.01 + T.zeros(disc_fake_D['c'].shape)).mean()
    LD_dG = bce(disc_fake_G['c'], T.ones(disc_fake_G['c'].shape)).mean()

    vae_loss = results_map['z_loss']
    rec_loss = compute_loss(x_reconstructed, normalize(x))

    loss = vae_loss + rec_loss

    inputs = [x]

    outputs = {'loss' : loss, 'vae_loss' : vae_loss, 'rec_loss' : rec_loss, 'reconstruction' : denormalize(x_reconstructed), 'c_real' : disc_real_D['c'], 'c_fake' : disc_fake_D['c'], 'x' : x, 'sample' : denormalize(x_sampled), 'interp' : denormalize(results_map['interp'])}

    print "params", params.keys()
    print "params enc", params_enc.keys()
    print "params dec", params_dec.keys()
    print "params_disc", params_disc.keys()

    updates = lasagne.updates.adam(LD_dG, params_dec.values(), learning_rate = 0.0001, beta1 = 0.5)
    updates_disc = lasagne.updates.adam(LD_dD + vae_loss * 0.0001, params_disc.values() + params_enc.values(), learning_rate = 0.0001, beta1 = 0.5)

    train_method = theano.function(inputs = inputs, outputs = outputs, updates = updates)
    disc_method = theano.function(inputs = inputs, outputs = outputs, updates = updates_disc)
    #gen_method = theano.function(inputs = inputs, outputs = outputs, updates = updates_gen)

    last_acc = 0.0

    for i in range(0,10000000):
        x = d.getBatch()

        t0 = time.time()
        res = train_method(x)
        disc_method(x)

        if i % 20 == 1:
            print "time", time.time() - t0

        if i % 100 == 1:
            d.saveExample(res['reconstruction'][0][:200], "image_reconstruction")
            d.saveExample(x[0][:200], "image_original")
            d.saveExample(res['sample'][0][:200],'image_sample')
            d.saveExampleWav(res['reconstruction'][0].astype('int16'), "image_reconstruction")
            d.saveExampleWav(x[0], "image_original")
            d.saveExampleWav(res['sample'][0].astype('int16'), "image_sample")
            
            d.saveExampleWav(res['interp'][[0,20,40,60,80,100,120],:].flatten().astype('int16'), "image_interp")

        if i % 20 == 1:
            print "==========================================================="
            print ""

            print "update", i, "loss", res['loss']
            print "vae loss", res['vae_loss']
            print "rec loss", res['rec_loss']
    
            print "classification", res['c_real'][:20].tolist()
            print res['c_fake'][:20].tolist()
            print "real", res['c_real'].mean(), "fake", res['c_fake'].mean()
            #print res['classification'].tolist()




