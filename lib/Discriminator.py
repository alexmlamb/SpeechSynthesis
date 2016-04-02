import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import DenseLayer
import lasagne

from layers import param_init_gru, gru_layer
from utils import init_tparams

bce = T.nnet.binary_crossentropy


class discriminator:

    def __init__(self, num_hidden, num_features, seq_length, mb_size, tf_states, rf_states):
        
        tf_states = T.specify_shape(tf_states, (seq_length, mb_size, num_features))
        rf_states = T.specify_shape(rf_states, (seq_length, mb_size, num_features))

        hidden_state_features = T.specify_shape(T.concatenate([tf_states, rf_states], axis = 1), (seq_length, mb_size * 2, num_features))

        gru_params_1 = init_tparams(param_init_gru(None, {}, prefix = "gru1", dim = num_hidden, nin = num_features))
        #gru_params_2 = init_tparams(param_init_gru(None, {}, prefix = "gru2", dim = num_hidden, nin = num_hidden + num_features))
        #gru_params_3 = init_tparams(param_init_gru(None, {}, prefix = "gru3", dim = num_hidden, nin = num_hidden + num_features))

        gru_1_out = gru_layer(gru_params_1, hidden_state_features, None, prefix = 'gru1')[0]
        #gru_2_out = gru_layer(gru_params_2, T.concatenate([gru_1_out, hidden_state_features], axis = 2), None, prefix = 'gru2', backwards = True)[0]
        #gru_3_out = gru_layer(gru_params_3, T.concatenate([gru_2_out, hidden_state_features], axis = 2), None, prefix = 'gru3')[0]

        final_out_recc = T.specify_shape(T.mean(gru_1_out, axis = 0), (mb_size * 2, num_hidden))

        h_out_1 = DenseLayer((mb_size * 2, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        #h_out_2 = DenseLayer((mb_size * 2, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        #h_out_3 = DenseLayer((mb_size * 2, num_hidden), num_units = num_hidden, nonlinearity=lasagne.nonlinearities.rectify)
        h_out_4 = DenseLayer((mb_size * 2, num_hidden), num_units = 1, nonlinearity=None)

        h_out_1_value = h_out_1.get_output_for(final_out_recc)
        h_out_4_value = h_out_4.get_output_for(h_out_1_value)

        raw_y = h_out_4_value
        #raw_y = T.clip(h_out_4_value, -10.0, 10.0)
        classification = T.nnet.sigmoid(raw_y)

        #tf comes before rf.  
        p_real =  classification[:mb_size]
        p_gen  = classification[mb_size:]

        #bce = lambda r,t: t * T.nnet.softplus(-r) + (1 - t) * (r + T.nnet.softplus(-r))

        self.d_cost_real = bce(p_real, 0.9 * T.ones(p_real.shape)).mean()
        self.d_cost_gen = bce(p_gen, 0.1 + T.zeros(p_gen.shape)).mean()
        self.g_cost_d = bce(p_gen, 0.9 * T.ones(p_gen.shape)).mean()
        self.d_cost = self.d_cost_real + self.d_cost_gen
        self.g_cost = self.g_cost_d


        self.classification = classification

        self.params = []
        self.params += lasagne.layers.get_all_params(h_out_4,trainable=True)
        #self.params += lasagne.layers.get_all_params(h_out_3,trainable=True)
        #self.params += lasagne.layers.get_all_params(h_out_2,trainable=True)
        self.params += lasagne.layers.get_all_params(h_out_1,trainable=True)

        self.params += gru_params_1.values()
        #self.params += gru_params_2.values()
        #self.params += gru_params_3.values()

        self.accuracy = T.mean(T.eq(T.ones(p_real.shape).flatten(), T.gt(p_real, 0.5).flatten())) + T.mean(T.eq(T.ones(p_gen.shape).flatten(), T.lt(p_gen, 0.5).flatten()))


       # self.accuracy = T.mean(T.eq(T.concatenate([T.ones(p_real.shape), T.zeros(p_gen.shape)], axis = 0), T.gt(classification, 0.5).flatten()))



