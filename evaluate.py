"""Evaluate a check-pointed model on MNIST test set.

E.g. python evaluate.py results/2016-12-14_11-45-25-basic
"""

import argparse
import numpy as np
import tensorflow as tf

from loss import elbo_loss
from datasets import binarized_mnist
from models import basic_decoder, basic_encoder, nf_encoder, iaf_encoder, hf_encoder, nn

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_dir')
args = parser.parse_args()

tf.reset_default_graph()

# Parse model name and flow
toks = args.checkpoint_dir.split('-')
model = toks[5]
flow = int(toks[6]) if len(toks) > 6 else None
batch_size = 100
print('model: %s\tflow:%s' % (model, flow))

# Build computation graph and operations
if model == 'basic':
    encoder_type = basic_encoder
if model == 'nf':
    encoder_type = nf_encoder
if model == 'iaf':
    encoder_type = iaf_encoder
if model == 'hf':
    encoder_type = hf_encoder

dim_x, dim_z, enc_dims, dec_dims = 784, 40, [300, 300], [300, 300]  # HACK
decoder_net = lambda z: nn(z, dec_dims, name='decoder', act=tf.nn.tanh)
encoder_net = lambda x: nn(x, enc_dims, name='encoder', act=tf.nn.tanh, is_training=False)
encoder = encoder_type(encoder_net, dim_z, flow)
decoder = basic_decoder(decoder_net, dim_x)
x = tf.placeholder(tf.float32, [None, dim_x], 'x')
x_w = tf.placeholder(tf.float32, [None, dim_x], 'x_w')
e = tf.placeholder(tf.float32, (None, dim_z), 'noise')
is_training = tf.placeholder(tf.bool)
z_params, z = encoder(x, e)
x_pred = decoder(z)
loss_ops = elbo_loss(x_pred, x, **z_params)

# Create a session and restore checkpoint
sess = tf.InteractiveSession()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
saver.restore(sess, ckpt.model_checkpoint_path)

# HACK: assumes specific directory structure and dataset
dataset = binarized_mnist()['test']
n_batches = dataset.num_examples / batch_size
feed_dict = {}
outputs = {k: [] for k in loss_ops.iterkeys()}

for _ in range(n_batches):
    feed_dict[x], feed_dict[x_w] = dataset.next_batch(batch_size, whitened=False)
    feed_dict[e] = np.random.normal(0, 1, (batch_size, dim_z))
    feed_dict[is_training] = False
    output = sess.run(loss_ops, feed_dict=feed_dict)
    for k, v in output.iteritems():
        outputs[k].append(v)

for k, vs in outputs.iteritems():
    outputs[k] = np.array(vs)
    print('avg. %s: %.3f' % (k, outputs[k].mean()))
