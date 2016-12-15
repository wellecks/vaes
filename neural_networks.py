"""Reusable neural network layers"""

import tensorflow as tf
import numpy as np

from nn_utils import *

def fc_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer."""
    input_dim = input_tensor.get_shape()[-1].value
    with tf.variable_scope(layer_name):
        with tf.variable_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        with tf.variable_scope('bias'):
            biases = bias_variable([output_dim])
        with tf.variable_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        if act is not None:
            activations = act(preactivate, name='activation')
        else:
            activations = preactivate
        return activations

def made_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu):

    def _get_made_masks(dim_in, dim_out):
        """See eqns. (8), (9) of Germain 2015. These assume a single hidden layer auto-encoder."""
        # msh[k] is max number of input units that the k'th hidden dimension can be connected to.
        msh = np.random.randint(1, dim_in, size=dim_out)
        # Eqn (8). An element is 1 when msh[k] >= d, for d in {1 ... dim_in}
        mask_in = (msh[:, np.newaxis] >= (np.tile(range(0, dim_in), [dim_out, 1]) + 1)).astype(np.float).T
        # Eqn (9). An element is 1 when d > msh[k]
        mask_out = ((np.tile(np.arange(0, dim_in)[:, np.newaxis], [1, dim_out])+1) > msh[np.newaxis, :]).astype(np.float).T
        return mask_in, mask_out

    input_dim = input_tensor.get_shape()[-1].value
    made_hidden_dim = 300
    with tf.variable_scope(layer_name):
        with tf.variable_scope('weight_in'):
            weights_in = weight_variable([input_dim, made_hidden_dim])
        with tf.variable_scope('weight_out'):
            weights_out = weight_variable([made_hidden_dim, input_dim])
        with tf.variable_scope('masks'):
            mask_in, mask_out = _get_made_masks(input_dim, made_hidden_dim)
        with tf.variable_scope('bias_in'):
            biases_in = bias_variable([made_hidden_dim])
        with tf.variable_scope('bias_out'):
            biases_out = bias_variable([input_dim])

        with tf.variable_scope('transformations'):
            hidden = tf.nn.relu(tf.matmul(input_tensor, weights_in * mask_in) + biases_in)
            preactivate = tf.matmul(hidden, weights_out * mask_out) + biases_out
            if act is not None:
                activations = act(preactivate, name='activation')
            else:
                activations = preactivate
        return activations

def nf_layer(input_tensor, output_dim, layer_name):
    # See equations (10), (11) of Kingma 2016
    input_dim = input_tensor.get_shape()[-1].value
    with tf.variable_scope(layer_name):
        with tf.variable_scope('u'):
            u = weight_variable(input_dim)
        with tf.variable_scope('w'):
            w = weight_variable(input_dim)
        with tf.variable_scope('b'):
            b = bias_variable(1)

        with tf.variable_scope('transformations'):
            z = input_tensor
            temp = tf.expand_dims(tf.nn.tanh(tf.reduce_sum(w * z, 1) + b), 1)
            temp = tf.tile(temp, [1, output_dim])
            z = z + tf.mul(u, temp)

            temp = tf.expand_dims(dtanh(tf.reduce_sum(w * z, 1) + b), 1)
            temp = tf.tile(temp, [1, output_dim])
            log_detj = tf.log(tf.abs(1. + tf.reduce_sum(tf.mul(u, temp * w), 1)))

        return z, log_detj

def nn(input_tensor, dims_hidden, name, act=tf.nn.relu, is_training=None):

    with tf.variable_scope(name):
        dim_out = dims_hidden[0]
        h = fc_layer(input_tensor, dim_out, 'layer0', act=None)
        if is_training is not None:
            h = tf.contrib.layers.batch_norm(inputs=h, decay=0.99, epsilon=1e-7, is_training=is_training)
        h = act(h)
        for i in range(len(dims_hidden) - 1):
            dim_in = dims_hidden[i]
            dim_out = dims_hidden[i+1]
            h = fc_layer(h, dim_out, 'layer_{}'.format(i+1), act=None)
            if is_training is not None:
                h = tf.contrib.layers.batch_norm(inputs=h, decay=0.99, epsilon=1e-7, is_training=is_training)
            h = act(h)
        dim_in = dims_hidden[-1]
    return h

def cnn(input_tensor, in_shape, layer_dict, output_dims_dict, name, act=tf.nn.tanh, final_act=None):
    with tf.variable_scope(name):
        hidden_channels, strides, filter_sizes = layer_dict['hidden_channels'], layer_dict['strides'], layer_dict['filter_sizes']
        h = conv_layer(input_tensor, in_shape[-1], hidden_channels[0], filter_sizes[0], strides[0], 'layer0', act)
        for i in range(len(hidden_channels) - 1):
            in_channels = hidden_channels[i]
            out_channels = hidden_channels[i+1]
            size = filter_sizes[i]
            stride = strides[i]
            h = conv_layer(h, in_channels, out_channels, size, stride, 'conv_layer_{}'.format(i+1), act)
        dim_flattened = np.prod(in_shape)
        #print dim_flattened
        h = tf.reshape(h, (-1, dim_flattened))
        #print h.get_shape()
        outputs = {}
        for key in output_dims_dict:
            outputs[key] = fc_layer(h, dim_flattened, output_dims_dict[key], layer_name=key, act=None)
        print [outputs[k].get_shape() for k in output_dims_dict]
    return outputs


# Create model
def conv_net(x, layer_dict):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    with tf.variable_scope('Conv1'):
        shape = layer_dict['wc1']
        weights, biases = weight_variable(shape), bias_variable(shape[-1])
        conv1 = conv2d(x, weights, biases)
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    with tf.variable_scope('Conv2'):
        shape = layer_dict['wc2']
        weights, biases = weight_variable(shape), bias_variable(shape[-1])
        conv2 = conv2d(conv1, weights, biases)
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    with tf.variable_scope('FC1'):
        shape = layer_dict['wd1']
        weights, biases = weight_variable(shape), bias_variable(shape[-1])
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.add(tf.matmul(fc1, weights), biases)
        fc1 = tf.nn.relu(fc1)

    # Output, class prediction
    with tf.variable_scope('Out'):
        shape = layer_dict['out']
        weights, biases = weight_variable(shape), bias_variable(shape[-1])
        out = tf.add(tf.matmul(fc1, weights), biases)
    return out
