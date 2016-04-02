import theano
import theano.tensor as T
import numpy as np
import Data
from layers import param_init_gru, gru_layer, param_init_fflayer, fflayer
import lasagne

import time

from Discriminator import discriminator

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

def get_generator_params(config):

    params = {}

    params = param_init_gru(options = None, param = params, prefix='gru1', nin=1, dim=config['num_hidden'])

    params = param_init_gru(options = None, param = params, prefix='gru2', nin=config['num_hidden'], dim=config['num_hidden'])

    params = param_init_fflayer(options = None, param = params, prefix='ff_1', nin=config['num_hidden'],nout=1,ortho=False)

    params = param_init_fflayer(options = None, param = params, prefix='ff_h1', nin=config['num_hidden'],nout=config['num_hidden'],ortho=False)

    params = param_init_fflayer(options = None, param = params, prefix='ff_h2', nin=config['num_hidden'],nout=config['num_hidden'],ortho=False)

    for paramKey in params:
        params[paramKey] = theano.shared(params[paramKey])

    return params

def rnn_one_step(config, params, observed_sequence_last, observed_sequence_current, use_samples, last_states, last_outputs, last_loss):

    mb_size = config['mb_size']
    num_hidden = config['num_hidden']

    last_states = T.specify_shape(last_states, (config['mb_size'],2 * config['num_hidden']))
    last_outputs = T.specify_shape(last_outputs, (config['mb_size'],))

    obs_last = T.specify_shape(observed_sequence_last, (mb_size,)).reshape((mb_size,1))
    obs_curr = T.specify_shape(observed_sequence_current, (mb_size,))

    obs_use = theano.ifelse.ifelse(use_samples, last_outputs.reshape((mb_size,1)), obs_last)

    last_states_1 = last_states[:,0:1024]
    last_states_2 = last_states[:,1024:2048]

    next_states_1 = T.specify_shape(gru_layer(params,state_below = obs_use, options = None, prefix='gru1', mask=None, one_step=True, init_state=last_states_1, backwards=False)[0], (mb_size, num_hidden))

    next_states_2 = T.specify_shape(gru_layer(params,state_below = next_states_1, options = None, prefix='gru2', mask=None, one_step=True, init_state=last_states_2, backwards=False)[0], (mb_size, num_hidden))

    h1 = T.specify_shape(fflayer(params,next_states_2,options=None,prefix='ff_h1',activ='lambda x: tensor.maximum(x,0.0)'), (mb_size, num_hidden))

    h2 = T.specify_shape(fflayer(params,h1,options=None,prefix='ff_h2',activ='lambda x: tensor.maximum(x,0.0)'), (mb_size, num_hidden))

    y = T.specify_shape(fflayer(params,h2,options = None,prefix='ff_1',activ='lambda x: x').flatten(), (mb_size,))
    #y = T.specify_shape(T.sum(next_states, axis = 1), (mb_size,))

    loss = T.sqr(y - obs_curr)

    obs_curr = T.specify_shape(observed_sequence_current, (mb_size,))

    next_outputs = y

    next_states = T.specify_shape(T.concatenate([next_states_1, next_states_2], axis = 1), (mb_size, num_hidden * 2))

    return next_states, next_outputs, loss



'''
Given:
    sequence: minibatch x time-steps
    do_sample: binary map.  1 = sampling, 0 = observed
    params: Just all the parameters to use.  

    Does a scan running forward for all the time steps, computing new hidden states and new outputs.  
    All hidden states are concatenated together.  


'''
def get_network(config, params, sequence, do_sample):

    mb_size = config['mb_size']
    seq_length = config['seq_length']
    num_hidden = config['num_hidden']

    sequence_ver = T.specify_shape(sequence * 1.0, (seq_length, mb_size))

    initial_states = theano.shared(np.zeros(shape = (mb_size, 2 * config['num_hidden'])).astype('float32'))
    initial_output = theano.shared(np.zeros(shape = (mb_size,)).astype('float32'))
    initial_loss = theano.shared(np.zeros(shape = (mb_size,)).astype('float32'))

    sequence_features = T.specify_shape(sequence_ver[:-1,:], (seq_length - 1, mb_size))
    sequence_target = T.specify_shape(sequence_ver[1:,:], (seq_length - 1, mb_size))

    use_samples = T.specify_shape(do_sample, (seq_length - 1,))

    results, _ = theano.scan(fn=lambda *inp: rnn_one_step(config, params, *inp), sequences=[sequence_features, sequence_target, use_samples], outputs_info=[initial_states, initial_output, initial_loss],non_sequences=[],n_steps = seq_length - 1)

    results[0] = T.specify_shape(results[0], (seq_length - 1, mb_size, 2 * num_hidden))
    results[1] = T.specify_shape(results[1], (seq_length - 1, mb_size))
    results[2] = T.specify_shape(results[2], (seq_length - 1, mb_size))

    return {'states' : results[0], 'output' : results[1], 'loss' : results[2]}

def normalize(inp):
    return inp / 1000.0

def denomalize(inp):
    return inp * 1000.0

if __name__ == "__main__":

    config = {}
    config['mb_size'] = 32
    config['seq_length'] = 256
    config['num_hidden'] = 1024

    data = Data.Data(mb_size = config['mb_size'], seq_length = config['seq_length'])

    seq = T.matrix()

    generator_params = get_generator_params(config)

    do_sample_TF = theano.shared(np.zeros(shape = (config['seq_length'] - 1,)).astype('int32'))
    do_sample_RF_100 = theano.shared(np.asarray([0.0] * 20 + [1.0] * (config['seq_length'] - 1 - 20)).astype('int32'))

    net = get_network(config, generator_params, normalize(seq.T), do_sample_TF)
    net_rf_100 = get_network(config, generator_params, normalize(seq.T), do_sample_RF_100)

    disc = discriminator(num_hidden = config['num_hidden'], num_features = 2 * config['num_hidden'], seq_length = config['seq_length'] - 1, mb_size = config['mb_size'], tf_states = net['states'], rf_states = net_rf_100['states'])

    for param in generator_params.values():
        print param.dtype
    for param in disc.params:
        print param.dtype

    updates = lasagne.updates.adam(T.mean(net['loss']), generator_params.values(), learning_rate = 0.001, beta1 = 0.9)

    updates_disc = lasagne.updates.adam(disc.d_cost, disc.params, learning_rate = 0.0001, beta1 = 0.5)

    updates.update(updates_disc)

    #f_disc = theano.function(inputs = [seq], outputs = {'acc' : disc.classification}, updates = updates_disc)

    f = theano.function(inputs = [seq], outputs = {'0' : net['states'], 'tf_output' : net['output'], 'rf_output' : net_rf_100['output'], 'tf_loss' : net['loss'], 'rf_loss' : net_rf_100['loss'], 'acc' : disc.classification}, updates = updates)

    for i in range(0, 800000):
        seq_val = data.getBatch()

        t0 = time.time()
        r = f(seq_val)
        print "mb time", time.time() - t0

        print "======================================="
        print i
        print "tf loss", r['tf_loss'][20:].mean()
        print "rf loss", r['rf_loss'][20:].mean()

        t0 = time.time()
        #rd = f_disc(seq_val)
        #print "d time", time.time() - t0

        rd = {'acc' : r['acc']}

        print "disc accuracy", rd['acc'][:16].mean(), rd['acc'][16:].mean()

        if i % 50 == 1:
            plt.figure(figsize = (16,8))
            plt.plot(r['tf_output'][:,0])
            plt.plot(r['rf_output'][:,0])
            plt.ylim(-3,3)
            plt.title("MB: " + str(i))
            #plt.plot(normalize(seq_val[0,:]), alpha = 0.5)
            plt.legend(["tf", "rf"])
            plt.savefig("plots/audio.png")
            plt.clf()


