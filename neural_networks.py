import tensorflow as tf
import numpy as np

from nn_utils import *
# We can't initialize these variables to 0 - the network will get stuck.

def fc_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses tanh to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    input_dim = input_tensor.get_shape()[-1].value
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.variable_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.variable_scope('weights'):
          weights = weight_variable([input_dim, output_dim])
          #variable_summaries(weights)
      with tf.variable_scope('bias'):
        biases = bias_variable([output_dim])
        #variable_summaries(biases)
      with tf.variable_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        #tf.histogram_summary('pre_activations', preactivate)
      if act is not None:
          activations = act(preactivate, name='activation')
      else: activations = preactivate
      #tf.histogram_summary('activations', activations)
      return activations

def made_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu):
    # Adding a name scope ensures logical grouping of the layers in the graph.

    def _get_made_masks(dim_in, dim_out):

        msh = np.random.randint(1, dim_in, size=dim_out)
        mask = (msh[:, np.newaxis] >= (np.tile(range(0, dim_in), [dim_out, 1])+1)).astype(np.float).T
        return mask

    input_dim = input_tensor.get_shape()[-1].value
    with tf.variable_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.variable_scope('in'):
            with tf.variable_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
            with tf.variable_scope('in_mask'):
                mask = _get_made_masks(input_dim, output_dim)
            with tf.variable_scope('bias'):
                biases = bias_variable([output_dim])
            #variable_summaries(biases)
            with tf.variable_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights * mask) + biases
            #tf.histogram_summary('pre_activations', preactivate)
            if act is not None:
                h = act(preactivate, name='activation')
            else: h = preactivate
        input_dim = h.get_shape()[-1].value
        with tf.variable_scope('out'):
            with tf.variable_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
            with tf.variable_scope('in_mask'):
                mask = _get_made_masks(input_dim, output_dim)
            with tf.variable_scope('bias'):
                biases = bias_variable([output_dim])
            with tf.variable_scope('Wx_plus_b'):
                preactivate = tf.matmul(h, weights * mask) + biases
            #tf.histogram_summary('pre_activations', preactivate)
            if act is not None:
                activations = act(preactivate, name='activation')
            else: activations = preactivate

        return activations

def nn(input_tensor, dims_hidden, name, act=tf.nn.relu):
    with tf.variable_scope(name):
        dim_out = dims_hidden[0]
        h = fc_layer(input_tensor, dim_out, 'layer0', act)
        for i in range(len(dims_hidden) - 1):
            dim_in = dims_hidden[i]
            dim_out = dims_hidden[i+1]
            h = fc_layer(h, dim_out, 'layer_{}'.format(i+1), act)
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
