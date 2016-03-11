import Data
import lasagne
import numpy as np

def init_params(config):
    params = {}

    params["W_enc_1"] = theano.shared(np.random.normal(size = (4000, 2048))
    params["W_enc_2"] = 

    params["b_enc_1"] = 
    params["b_enc_2"] = 

    params["z_mean_W"] = 
    params["z_mean_b"] = 

    params["z_std_W"] =
    params["z_std_b"] = 

    params['W_dec_1'] = 
    params['W_dec_2'] = 
    params['W_dec_3'] = 

    params['b_dec_1'] = 
    params['b_dec_2'] = 
    params['b_dec_3'] = 

    return params

'''
Maps from a given x to an h_value.  


'''
def encoder(params):
    pass


def decoder(params):
    pass



def define_network(x):
    pass


if __name__ == "__main__":



