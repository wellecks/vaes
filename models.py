
import tensorflow as tf
from utils import *
import numpy as np
import pdb
###### ENCODERS

def basic_encoder(x, e, dim_x, dims_hidden, dim_z):
    output_dims_dict = {'mu': dim_z, 'log_std': dim_z}
    last_hidden = nn(x, dim_x, dims_hidden, 'encoder', act=tf.nn.tanh)
    outputs = {}
    for key in output_dims_dict:
        outputs[key] = fc_layer(last_hidden, dims_hidden[-1], output_dims_dict[key], layer_name=key, act=None)
    mu, log_std = outputs['mu'], outputs['log_std']
    z = mu + tf.exp(log_std) * e
    return outputs, z

def conv_encoder(layer_dict, in_shape):
    def _conv_encoder(x, e, in_shape, dim_z, layer_dict, width):
        reshaped = tf.reshape(x, (-1, width, width, 1))
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z}
        outputs = cnn(reshaped, in_shape, layer_dict, output_dims_dict, 'encoder', act=tf.nn.tanh, final_act=None)
        mu, log_std = outputs['mu'], outputs['log_std']
        z = mu + tf.exp(log_std) * e
        return outputs, z
    return lambda x, e, dim_z, width: _conv_encoder(x, e, in_shape, dim_z, layer_dict, width)

def nf_encoder(flow):
    def _nf_encoder(x, e, dim_x, dims_hidden, dim_z, flow):
        def norm_flow(z, us, ws, bs, flow):
            sum_log_detj = 0
            for k in range(flow):
                u, w, b = us[:, k*dim_z:(k+1)*dim_z], ws[:, k*dim_z:(k+1)*dim_z], bs[:, k]
                temp = tf.expand_dims(tf.nn.tanh(tf.reduce_sum(w * z, 1) + b), 1)
                temp = tf.tile(temp, [1, u.get_shape()[1].value])
                z = z + tf.mul(u, temp)

                # Eqn. (11) and (12)
                temp = tf.expand_dims(dtanh(tf.reduce_sum(w * z, 1) + b), 1)
                psi = tf.tile(temp, [1, w.get_shape()[1].value]) * w
                log_detj = tf.abs(1. + tf.reduce_sum(tf.mul(u, psi), 1))
                sum_log_detj += log_detj
            return z, sum_log_detj

        output_dims_dict = {'mu': dim_z, 'log_std': dim_z, 'us': dim_z * flow, 'ws': dim_z * flow, 'bs': dim_z * flow}
        last_hidden = nn(x, dim_x, dims_hidden, 'encoder', act=tf.nn.tanh)
        outputs = {}
        for key in output_dims_dict:
            outputs[key] = fc_layer(last_hidden, dims_hidden[-1], output_dims_dict[key], layer_name=key, act=None)

        mu, log_std, us, ws, bs = outputs['mu'], outputs['log_std'], outputs['us'], outputs['ws'], outputs['bs']

        z0 = mu + tf.exp(log_std) * e
        outputs['z0'] = z0 # this is z0, pre-flow
        z, sum_log_detj = norm_flow(z0, us, ws, bs, flow)
        outputs['zk'] = z # this is zk, post-flow
        outputs['sum_log_detj'] = sum_log_detj
        return outputs, z

    return lambda x, e, dim_x, dims_hidden, dim_z: _nf_encoder(x, e, dim_x, dims_hidden, dim_z, flow)

def iaf_encoder(flow):
    def _iaf_encoder(x, e, dim_x, dims_hidden, dim_z, flow):
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z, 'flow_mus': dim_z * flow, 'flow_sds': dim_z * flow}
        last_hidden = nn(x, dim_x, dims_hidden, 'encoder', act=tf.nn.tanh)
        outputs = {}
        for key in ['mu', 'log_std']:
            outputs[key] = fc_layer(last_hidden, dims_hidden[-1], output_dims_dict[key], layer_name=key, act=None)
        for key in ['flow_mus', 'flow_sds']:
            outputs[key] = made_layer(last_hidden, dims_hidden[-1], output_dims_dict[key], layer_name=key, act=None)

        mu, log_std, flow_mus, flow_sds = outputs['mu'], outputs['log_std'], outputs['flow_mus'], outputs['flow_sds']

        z0 = mu + tf.exp(log_std) * e # preflow

        z, log_detj = _iaf(z0, flow_mus, flow_sds) # apply the IAF
        outputs['log_detj'] = log_detj
        return outputs, z
    return lambda x, e, dim_x, dims_hidden, dim_z: _iaf_encoder(x, e, dim_x, dims_hidden, dim_z, flow)

def made(input_data, Z, K, name, act_fn):
    with tf.variable_scope('made'):
        # Follows equations (6) and (7) of [Germain 2015]
        W, b, V, c = _init_made_params(Z, K, name)
        M_W, M_V = _create_made_masks(Z, K, name)
        h = tf.nn.relu(b + tf.matmul(input_data, W * M_W))
        out = act_fn(c + tf.matmul(h, (V * M_V)), name='%s_out' % name)
    return out

def _init_made_params(D, K, name, initializer=tf.contrib.layers.xavier_initializer):
    with tf.variable_scope('%s' % name):
        W = tf.get_variable('W', [D, K], initializer=initializer())
        b = tf.get_variable('b', [K])
        V = tf.get_variable('V', [K, D], initializer=initializer())
        c = tf.get_variable('c', [D])
    return W, b, V, c

def _create_made_masks(dim_in, dim_out):
    with tf.variable_scope('%s' % name):
        # See equations (8) and (9) of [Germain 2015]

        msh = np.random.randint(1, dim_in, size=dim_out)
        M_W_ = (msh[:, np.newaxis] >= (np.tile(range(0, dim_in), [dim_out, 1])+1)).astype(np.float).T
        M_V_ = ((np.tile(np.arange(0, dim_in)[:, np.newaxis], [1, dim_out])+1) > msh[np.newaxis, :]).astype(np.float).T
        M_W = tf.constant(M_W_, 'float32', M_W_.shape, name='M_W')
        M_V = tf.constant(M_V_, 'float32', M_V_.shape, name='M_V')
    return M_W, M_V

def get_made_mask(dim_in, dim_out):
    msh = np.random.randint(1, dim_in, size=dim_out)
    mask = (msh[:, np.newaxis] >= (np.tile(range(0, dim_in), [dim_out, 1])+1)).astype(np.float).T
    return mask, msh

def inverse_autoregressive_flow(z, Z, K):
    made_mu = made(z, Z, K, 'made_mu', tf.nn.sigmoid)
    made_sd = made(z, Z, K, 'made_sd', tf.nn.sigmoid)

    # IAF transformation and log det of Jacobian; eqns (9) and (10) of [Kingma 2016]
    z = (z - made_mu) / made_sd
    log_detj = -tf.reduce_sum(tf.log(made_sd), 1)
    return z, log_detj

def _iaf(z, mu, sd):
    # IAF transformation and log det of Jacobian; eqns (9) and (10) of [Kingma 2016]
    z = (z - mu) #TODO currently mu has shape (dim_z, dim_z) whereas z has shape (bs, dim_z)
    z /= sd
    log_detj = -tf.reduce_sum(tf.log(sd), 1)
    return z, log_detj

###### DECODERS
def basic_decoder(z, dim_x, dims_hidden, dim_z):
    with tf.variable_scope('decoder'):
        dim_out = dims_hidden[0]
        h = fc_layer(z, dim_z, dim_out, 'layer0')
        for i in range(len(dims_hidden) - 1):
            dim_in = dims_hidden[i]
            dim_out = dims_hidden[i+1]
            h = fc_layer(h, dim_in, dim_out, 'layer_{}'.format(i+1))
        dim_in = dims_hidden[-1]
        x_pred = fc_layer(h, dim_in, dim_x, 'mu', act=tf.sigmoid)
    return x_pred

#####################################
# DEPRECATED
#####################################
def old_encoder(x, e, D, H, Z, initializer=tf.contrib.layers.xavier_initializer):
    with tf.variable_scope('encoder'):
        w_h = tf.get_variable('w_h', [D, H], initializer=initializer())
        b_h = tf.get_variable('b_h', [H])
        w_mu = tf.get_variable('w_mu', [H, Z], initializer=initializer())
        b_mu = tf.get_variable('b_mu', [Z])
        w_v = tf.get_variable('w_v', [H, Z], initializer=initializer())
        b_v = tf.get_variable('b_v', [Z])

        h = tf.nn.tanh(tf.matmul(x, w_h) + b_h)
        mu = tf.matmul(h, w_mu) + b_mu
        log_var = tf.matmul(h, w_v) + b_v
        z = mu + tf.sqrt(tf.exp(log_var))*e
    return mu, log_var, z

def old_decoder(z, D, H, Z, initializer=tf.contrib.layers.xavier_initializer):
    with tf.variable_scope('decoder'):
        w_h = tf.get_variable('w_h', [Z, H], initializer=initializer())
        b_h = tf.get_variable('b_h', [H])
        w_mu = tf.get_variable('w_mu', [H, D], initializer=initializer())
        b_mu = tf.get_variable('b_mu', [D])
        w_v = tf.get_variable('w_v', [H, 1], initializer=initializer())
        b_v = tf.get_variable('b_v', [1])

        h = tf.nn.tanh(tf.matmul(z, w_h) + b_h)
        out_mu = tf.matmul(h, w_mu) + b_mu
        out_log_var = tf.matmul(h, w_v) + b_v
        # NOTE(wellecks) Enforce 0, 1 (MNIST-specific)
        out = tf.sigmoid(out_mu)
    return out, out_mu, out_log_var
