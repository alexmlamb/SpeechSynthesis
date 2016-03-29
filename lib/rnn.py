import theano
import theano.tensor as T
import numpy as np
import Data

def get_generator_params():

    params = {}


    return params

def rnn_one_step(config, params, observed_sequence_last, observed_sequence_current, last_states, last_outputs):

    last_states = T.specify_shape(last_states, (config['mb_size'],config['num_hidden']))
    last_outputs = T.specify_shape(last_outputs, (config['mb_size'],))

    obs = T.specify_shape(observed_sequence_last, (128,))
    obs2 = T.specify_shape(observed_sequence_current, (128,))

    next_states = last_states
    next_outputs = last_outputs

    return next_states, next_outputs + obs + obs2



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

    initial_states = theano.shared(np.zeros(shape = (mb_size, 1024)).astype('float32'))
    initial_output = theano.shared(np.zeros(shape = (mb_size,)).astype('float32'))

    sequence_features = T.specify_shape(sequence_ver[:-1,:], (seq_length - 1, mb_size))
    sequence_target = T.specify_shape(sequence_ver[1:,:], (seq_length - 1, mb_size))

    results, _ = theano.scan(fn=lambda *inp: rnn_one_step(config, params, *inp), sequences=[sequence_features, sequence_target], outputs_info=[initial_states, initial_output],non_sequences=[],n_steps = seq_length - 1)

    results[0] = T.specify_shape(results[0], (seq_length - 1, mb_size, num_hidden))
    results[1] = T.specify_shape(results[1], (seq_length - 1, mb_size))

    return results

if __name__ == "__main__":

    config = {}
    config['mb_size'] = 128
    config['seq_length'] = 256
    config['num_hidden'] = 1024

    data = Data.Data(mb_size = config['mb_size'], seq_length = config['seq_length'])

    seq = T.matrix()

    generator_params = get_generator_params()

    net = get_network(config, generator_params, seq.T, None)

    f = theano.function(inputs = [seq], outputs = {'0' : net[0], '1' : net[1]})

    seq_val = data.getBatch()


    r = f(seq_val)
    print r['0'].shape
    print r['1'].shape



