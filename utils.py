import tensorflow as tf
import numpy as np

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    return tf.get_variable('weights', shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    return tf.get_variable('bias', shape, initializer=tf.constant_initializer(0.1))

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.variable_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.scalar_summary('mean', mean)
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.scalar_summary('stddev', stddev)
      tf.scalar_summary('max', tf.reduce_max(var))
      tf.scalar_summary('min', tf.reduce_min(var))
      tf.histogram_summary('histogram', var)

def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses tanh to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
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

def dtanh(tensor):
    return 1.0 - tf.square(tf.tanh(tensor))

def conv_layer(input_tensor, in_channels, out_channels, filter_size, stride, layer_name, act=tf.nn.relu):
    with tf.variable_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.variable_scope('weights'):
          weights = weight_variable([filter_size, filter_size, in_channels, out_channels])
          #variable_summaries(weights)
      with tf.variable_scope('bias'):
        biases = bias_variable([1, 1, out_channels])
        #variable_summaries(biases)
      with tf.variable_scope('Wx_plus_b'):
        preactivate = tf.nn.conv2d(input_tensor, filter=weights, strides=[1, stride, stride, 1], padding='SAME') + biases
        #tf.histogram_summary('pre_activations', preactivate)
      if act is not None:
          activations = act(preactivate, name='activation')
      else: activations = preactivate
      #tf.histogram_summary('activations', activations)
      return activations

def made_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # Adding a name scope ensures logical grouping of the layers in the graph.

    def _get_made_mask(dim_in, dim_out):
        msh = np.random.randint(1, dim_in, size=dim_out)
        mask = (msh[:, np.newaxis] >= (np.tile(range(0, dim_in), [dim_out, 1])+1)).astype(np.float).T
        return mask

    with tf.variable_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.variable_scope('weights'):
          weights = weight_variable([input_dim, output_dim])
      with tf.variable_scope('mask'):
          mask = _get_made_mask(input_dim, output_dim)
          #variable_summaries(weights)
      with tf.variable_scope('bias'):
        biases = bias_variable([output_dim])
        #variable_summaries(biases)
      with tf.variable_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights * mask) + biases
        #tf.histogram_summary('pre_activations', preactivate)
      if act is not None:
          activations = act(preactivate, name='activation')
      else: activations = preactivate
      #tf.histogram_summary('activations', activations)
      return activations

def nn(input_tensor, dim_start, dims_hidden, name, act=tf.nn.relu):
    with tf.variable_scope(name):
        dim_out = dims_hidden[0]
        h = fc_layer(input_tensor, dim_start, dim_out, 'layer0', act)
        for i in range(len(dims_hidden) - 1):
            dim_in = dims_hidden[i]
            dim_out = dims_hidden[i+1]
            h = fc_layer(h, dim_in, dim_out, 'layer_{}'.format(i+1), act)
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

def crossentropy(obs, actual, offset=1e-7):
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) +
                                  (1 - actual) * tf.log(1 - obs_), 1)
