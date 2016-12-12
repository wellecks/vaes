
import tensorflow as tf
from neural_networks import *
import numpy as np
import pdb
###### ENCODERS

def basic_encoder(neural_net, dim_z):
    def _basic_encoder(x, e, neural_net, dim_z):
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z}
        last_hidden = neural_net(x)
        outputs = {}
        for key in output_dims_dict:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)
        mu, log_std = outputs['mu'], outputs['log_std']
        z = mu + tf.exp(log_std) * e
        return outputs, z
    return lambda x, e: _basic_encoder(x, e, neural_net, dim_z)

def nf_encoder(neural_net, dim_z, flow):
    def _nf_encoder(x, e, neural_net, dim_z, flow):
        def norm_flow_one_step(z, u, w, b):
            temp = tf.expand_dims(tf.nn.tanh(tf.reduce_sum(w * z, 1) + b), 1)
            temp = tf.tile(temp, [1, u.get_shape()[1].value])
            z = z + tf.mul(u, temp)

            # Eqn. (11) and (12)
            temp = tf.expand_dims(dtanh(tf.reduce_sum(w * z, 1) + b), 1)
            psi = tf.tile(temp, [1, w.get_shape()[1].value]) * w
            log_detj = tf.abs(1. + tf.reduce_sum(tf.mul(u, psi), 1))
            return z, log_detj

        def norm_flow(z, us, ws, bs):
            d_z = z.get_shape()[-1].value # Get dimension of z
            K = us.get_shape()[-1].value / d_z # Find length of flow from given parameters
            sum_log_detj = 0.0
            for k in range(K):
                u, w, b = us[:, k*dim_z:(k+1)*d_z], ws[:, k*d_z:(k+1)*d_z], bs[:, k]
                z, log_detj = norm_flow_one_step(z, u, w, b)
                sum_log_detj += log_detj
            return z, sum_log_detj

        output_dims_dict = {'mu': dim_z, 'log_std': dim_z, 'us': dim_z * flow, 'ws': dim_z * flow, 'bs': dim_z * flow}
        last_hidden = neural_net(x)
        outputs = {}
        for key in output_dims_dict:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)

        mu, log_std, us, ws, bs = outputs['mu'], outputs['log_std'], outputs['us'], outputs['ws'], outputs['bs']

        z0 = mu + tf.exp(log_std) * e
        zk, sum_log_detj = norm_flow(z0, us, ws, bs)

        outputs['z0'] = z0 # this is z0, pre-flow
        outputs['zk'] = zk # this is zk, post-flow
        outputs['sum_log_detj'] = sum_log_detj
        return outputs, zk

    return lambda x, e: _nf_encoder(x, e, neural_net, dim_z, flow)

def iaf_encoder(neural_net, dim_z, flow):
    def _iaf_encoder(x, e, neural_net, dim_z, flow):
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z, 'flow_mus': dim_z * flow, 'flow_log_stds': dim_z * flow}
        last_hidden = neural_net(x)
        outputs = {}
        for key in ['mu', 'log_std']:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)
        extra_hidden = fc_layer(last_hidden, dim_z, layer_name='extra_hidden', act=None)

        for key in ['flow_mus', 'flow_log_stds']:
            outputs[key] = made_layer(extra_hidden, output_dims_dict[key], layer_name=key)

        mu, log_std, flow_mus, flow_log_stds = outputs['mu'], outputs['log_std'], outputs['flow_mus'], outputs['flow_log_stds']

        z0 = mu + tf.exp(log_std) * e # preflow

        flow_stds = tf.exp(flow_log_stds)
        zk, sum_log_detj = inverse_autoregressive_flow(z0, flow_mus, flow_stds) # apply the IAF
        outputs['sum_log_detj'] = sum_log_detj
        outputs['z0'] = z0
        outputs['zk'] = zk

        return outputs, zk

    def inverse_autoregressive_flow_one_step(z, mu, std):
        z = (z - mu) / std
        log_detj = -tf.reduce_sum(tf.log(std), 1)
        return z, log_detj

    def inverse_autoregressive_flow(z, mus, stds):
        d_z = z.get_shape()[-1].value # Get dimension of z
        K = mus.get_shape()[-1].value / d_z # Find length of flow from given parameters
        # IAF transformation and log det of Jacobian; eqns (9) and (10) of [Kingma 2016]
        sum_log_detj = 0.0
        for k in range(K):
            mu, std = mus[:, k*d_z:(k+1)*d_z], stds[:, k*d_z:(k+1)*d_z]
            z, log_detj = inverse_autoregressive_flow_one_step(z, mu, std)
            sum_log_detj += log_detj
        return z, sum_log_detj

    return lambda x, e: _iaf_encoder(x, e, neural_net, dim_z, flow)

def hf_encoder(neural_net, dim_z, flow):
    def _hf_encoder(x, e, neural_net, dim_z, flow):
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z, 'flow_vs': dim_z * flow}
        last_hidden = neural_net(x)
        outputs = {}
        for key in ['mu', 'log_std']:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)
        if output_dims_dict['flow_vs'] != 0:
            outputs['flow_vs'] = fc_layer(last_hidden, output_dims_dict['flow_vs'], layer_name='flow_vs', act=None)
        else: outputs['flow_vs'] = None
        mu, log_std, flow_vs = outputs['mu'], outputs['log_std'], outputs['flow_vs']

        z0 = mu + tf.exp(log_std) * e # preflow

        zk, sum_log_detj = householder_flow(z0, flow_vs) # apply the IAF
        outputs['sum_log_detj'] = sum_log_detj
        outputs['z0'] = z0
        outputs['zk'] = zk

        return outputs, zk

    def householder_flow(z, vs):
        if vs is not None:
            d_z = z.get_shape()[-1].value # Get dimension of z
            K = vs.get_shape()[-1].value / d_z # Find length of flow from given parameters
            for k in range(K):
                v = vs[:, k*d_z:(k+1)*d_z]
                z = householder(z, v)
        sum_log_detj = 0.0
        return z, sum_log_detj

    def householder(z, v):
        norm_squared_v = tf.expand_dims(tf.reduce_sum(tf.pow(v, 2), 1, keep_dims=True), 1) # HACKHACKHACK
        H = 1 - 2 * tf.mul(tf.expand_dims(v, 1), tf.expand_dims(v, 2)) / norm_squared_v
        return tf.reduce_sum(tf.mul(H, tf.expand_dims(z, 1)), 2)

    return lambda x, e: _hf_encoder(x, e, neural_net, dim_z, flow)

###### DECODERS
def basic_decoder(neural_net, dim_x):
    def _basic_decoder(z, neural_net, dim_x):
        with tf.variable_scope('decoder'):
            last_hidden = neural_net(z)
            x_pred = fc_layer(last_hidden, dim_x, 'mu', act=tf.sigmoid)
        return x_pred
    return lambda z: _basic_decoder(z, neural_net, dim_x)

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


def old_basic_encoder(dim_x, dims_hidden, dim_z):
    def _old_basic_encoder(x, e, dim_x, dims_hidden, dim_z):
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z}
        last_hidden = nn(x, dim_x, dims_hidden, 'encoder', act=tf.nn.tanh)
        outputs = {}
        for key in output_dims_dict:
            outputs[key] = fc_layer(last_hidden, dims_hidden[-1], output_dims_dict[key], layer_name=key, act=None)
        mu, log_std = outputs['mu'], outputs['log_std']
        z = mu + tf.exp(log_std) * e
        return outputs, z
    return lambda x, e: _old_basic_encoder(x, e, dim_x, dims_hidden, dim_z)

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
