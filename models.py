"""Encoder and Decoder implementations"""

import numpy as np
from neural_networks import *

def basic_encoder(neural_net, dim_z, *args):
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

def nf_encoder(neural_net, dim_z, flow, use_c=True):
    def _nf_encoder(x, e, neural_net, dim_z, flow, use_c):

        def norm_flow_one_step(z, u, w, b):
            temp = tf.nn.tanh(tf.reduce_sum(w * z, 1, keep_dims=True) + b)
            z = z + tf.mul(u, temp)

            # Eqn. (11) and (12)
            temp = dtanh(tf.reduce_sum(w * z, 1, keep_dims=True) + b)
            psi = temp * w
            log_detj = tf.log(tf.abs(1. + tf.reduce_sum(tf.mul(u, psi), 1)))
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

        def get_norm_flow_params(c, use_c):
            #shape = z.get_shape()[0].value
            #v = tf.get_variable('constant_term', shape, initializer=tf.constant_initializer(0.0), trainable=False)
            v = tf.ones_like(c)
            if use_c:
                v = tf.concat(1, (v, c))

            u = fc_layer(v, dim_z, layer_name='u', act=None)
            w = fc_layer(v, dim_z, layer_name='w', act=None)
            b = fc_layer(v, 1, layer_name='b', act=None)
            return u, w, b

        def nf(z, c, use_c, flow_length):
            z = z0
            sum_log_detj = 0.0
            for i in range(flow_length):
                with tf.variable_scope('flow_{}'.format(i)):
                    u, w, b = get_norm_flow_params(c, use_c)
                    z, log_detj = norm_flow_one_step(z, u, w, b)
                    sum_log_detj += log_detj

            return z, sum_log_detj

        #output_dims_dict = {'mu': dim_z, 'log_std': dim_z, 'us': dim_z * flow, 'ws': dim_z * flow, 'bs': dim_z * flow}
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z}

        last_hidden = neural_net(x)
        outputs = {}
        for key in ['mu', 'log_std']:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)

        #mu, log_std, us, ws, bs = outputs['mu'], outputs['log_std'], outputs['us'], outputs['ws'], outputs['bs']
        mu, log_std = outputs['mu'], outputs['log_std']

        z0 = mu + tf.exp(log_std) * e

        #zk, sum_log_detj = norm_flow(z0, us, ws, bs)
        zk, sum_log_detj = nf(z0, last_hidden, use_c, flow_length=flow)

        outputs['sum_log_detj'] = sum_log_detj
        outputs['z0'] = z0
        outputs['zk'] = zk

        return outputs, zk

    return lambda x, e: _nf_encoder(x, e, neural_net, dim_z, flow, use_c)

def iaf_encoder(neural_net, dim_z, flow):
    def _iaf_encoder(x, e, neural_net, dim_z, flow):
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z}
        flow_dims_dict = {'mu': dim_z * flow, 'log_std': dim_z * flow}

        last_hidden = neural_net(x)
        outputs = {}
        for key in ['mu', 'log_std']:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)

        mu, log_std = outputs['mu'], outputs['log_std']
        z0 = mu + tf.exp(log_std) * e  # preflow
        zk, sum_log_detj = iaf(z0, flow, flow_dims_dict['mu'], flow_dims_dict['log_std'])

        outputs['sum_log_detj'] = sum_log_detj
        outputs['z0'] = z0
        outputs['zk'] = zk

        return outputs, zk

    def iaf(z0, K, mu_dim, log_std_dim):
        z = z0
        log_detj = 0.0
        for k in range(K):
            # See equations (10), (11) of Kingma 2016
            mu = made_layer(z, mu_dim, 'flow_mu_%d' % k)
            log_std = made_layer(z, log_std_dim, 'flow_log_std_%d' % k)
            z = (z - mu) / tf.exp(log_std)
            log_detj += -tf.reduce_sum(log_std, 1)
        return z, log_detj

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
            d_z = z.get_shape()[-1].value  # Get dimension of z
            K = vs.get_shape()[-1].value / d_z  # Find length of flow from given parameters
            for k in range(K):
                v = vs[:, k*d_z:(k+1)*d_z]
                z = householder(z, v)
        sum_log_detj = 0.0
        return z, sum_log_detj

    def householder(z, v):
        norm_squared_v = tf.expand_dims(tf.reduce_sum(tf.pow(v, 2), 1, keep_dims=True), 1)  # HACK

        I = tf.constant(np.identity(dim_z, dtype=np.float32))
        H = I - 2 * tf.mul(tf.expand_dims(v, 1), tf.expand_dims(v, 2)) / norm_squared_v
        return tf.reduce_sum(tf.mul(H, tf.expand_dims(z, 1)), 2)

    return lambda x, e: _hf_encoder(x, e, neural_net, dim_z, flow)

def linear_iaf_encoder(neural_net, dim_z, *args):
    def _linear_iaf_encoder(x, e, neural_net, dim_z):
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z, 'L':dim_z * dim_z}
        last_hidden = neural_net(x)
        outputs = {}
        for key in output_dims_dict:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)
        mu, log_std, L = outputs['mu'], outputs['log_std'], outputs['L']
        mask = tf.expand_dims(tf.constant(np.triu(np.zeros((dim_z, dim_z))), dtype=tf.float32), 0)
        L = tf.reshape(L, (-1, dim_z, dim_z))
        temp = mask * L
        ones = tf.expand_dims(tf.constant(np.eye(dim_z), dtype=tf.float32), 0)
        L = temp + ones
        z0 = mu + tf.exp(log_std) * e

        zk = tf.reduce_sum(tf.mul(L, tf.expand_dims(z0, 1)), 2)
        outputs['sum_log_detj'] = 0.0
        outputs['z0'] = z0
        outputs['zk'] = zk
        return outputs, zk
    return lambda x, e: _linear_iaf_encoder(x, e, neural_net, dim_z)

# DECODERS
def basic_decoder(neural_net, dim_x, act=tf.sigmoid):
    def _basic_decoder(z, neural_net, dim_x):
        with tf.variable_scope('decoder'):
            last_hidden = neural_net(z)
            x_pred = fc_layer(last_hidden, dim_x, 'mu', act=act)
        return x_pred
    return lambda z: _basic_decoder(z, neural_net, dim_x)
