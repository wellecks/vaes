
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
        last_hidden = neural_net(x)
        outputs = {}
        for key in output_dims_dict:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)

        mu, log_std, us, ws, bs = outputs['mu'], outputs['log_std'], outputs['us'], outputs['ws'], outputs['bs']

        z0 = mu + tf.exp(log_std) * e
        zk, sum_log_detj = norm_flow(z0, us, ws, bs, flow)

        outputs['z0'] = z0 # this is z0, pre-flow
        outputs['zk'] = zk # this is zk, post-flow
        outputs['sum_log_detj'] = sum_log_detj
        return outputs, zk

    return lambda x, e: _nf_encoder(x, e, neural_net, dim_z, flow)

def iaf_encoder(neural_net, dim_z, flow):
    def _iaf_encoder(x, e, neural_net, dim_z, flow):
        output_dims_dict = {'mu': dim_z, 'log_std': dim_z, 'flow_mus': dim_z * flow, 'flow_sds': dim_z * flow}
        last_hidden = neural_net(x)
        outputs = {}
        for key in ['mu', 'log_std']:
            outputs[key] = fc_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)
        for key in ['flow_mus', 'flow_sds']:
            outputs[key] = made_layer(last_hidden, output_dims_dict[key], layer_name=key, act=None)

        mu, log_std, flow_mus, flow_sds = outputs['mu'], outputs['log_std'], outputs['flow_mus'], outputs['flow_sds']

        z0 = mu + tf.exp(log_std) * e # preflow

        zk, sum_log_detj = _iaf(z0, flow_mus, flow_sds, flow) # apply the IAF
        outputs['sum_log_detj'] = sum_log_detj
        outputs['z0'] = z0
        outputs['zk'] = zk

        return outputs, zk

    def _iaf(z, mus, sds, flow):
        # IAF transformation and log det of Jacobian; eqns (9) and (10) of [Kingma 2016]
        sum_log_detj = 0
        for k in range(flow):
            mu, sd = mus[:, k*dim_z:(k+1)*dim_z], sds[:, k*dim_z:(k+1)*dim_z]
            z = (z - mu) #TODO currently mu has shape (dim_z, dim_z) whereas z has shape (bs, dim_z)
            z /= sd
            sum_log_detj += -tf.reduce_sum(tf.log(sd), 1)
        return z, sum_log_detj

    return lambda x, e: _iaf_encoder(x, e, neural_net, dim_z, flow)


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
