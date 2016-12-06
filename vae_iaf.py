"""Variational Auto-Encoder with Inverse Autoregressive Flow (using MADE)

References
----------
https://arxiv.org/pdf/1606.04934v1.pdf
https://arxiv.org/pdf/1502.03509v2.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import *

def inputs(D, Z):
    x = tf.placeholder(tf.float32, [None, D], 'x')
    e = tf.placeholder(tf.float32, [None, Z], 'e')
    return x, e

def encoder(x, e, D, H, Z, K, initializer=tf.contrib.layers.xavier_initializer):
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

def decoder(z, D, H, Z, initializer=tf.contrib.layers.xavier_initializer, out_fn=tf.sigmoid):
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
        out = out_fn(out_mu)
    return out, out_mu, out_log_var

def made(input_data, Z, H, name, act_fn):
    with tf.variable_scope('made'):
        # Follows equations (6) and (7) of [Germain 2015]
        W, b, V, c = _init_made_params(Z, H, name)
        M_W, M_V = _create_made_masks(Z, H, name)
        h = tf.nn.relu(b + tf.matmul(input_data, W*M_W))
        out = act_fn(c + tf.matmul(h, (V*M_V)), name='%s_out' % name)
    return out

def _init_made_params(D, K, name, initializer=tf.contrib.layers.xavier_initializer):
    with tf.variable_scope('%s' % name):
        W = tf.get_variable('W', [D, K], initializer=initializer())
        b = tf.get_variable('b', [K])
        V = tf.get_variable('V', [K, D], initializer=initializer())
        c = tf.get_variable('c', [D])
    return W, b, V, c

def _create_made_masks(D, K, name):
    with tf.variable_scope('%s' % name):
        # See equations (8) and (9) of [Germain 2015]
        msh = np.random.randint(1, D, size=K)
        M_W_ = (msh[:, np.newaxis] >= (np.tile(range(0, D), [K, 1])+1)).astype(np.float).T
        M_V_ = ((np.tile(np.arange(0, D)[:, np.newaxis], [1, K])+1) > msh[np.newaxis, :]).astype(np.float).T
        M_W = tf.constant(M_W_, 'float32', M_W_.shape, name='M_W')
        M_V = tf.constant(M_V_, 'float32', M_V_.shape, name='M_V')
    return M_W, M_V

def inverse_autoregressive_flow(z, Z, H):
    made_mu = made(z, Z, H, 'made_mu', tf.nn.sigmoid)
    made_sd = made(z, Z, H, 'made_sd', tf.nn.sigmoid)

    # IAF transformation and log det of Jacobian; eqns (9) and (10) of [Kingma 2016]
    z = (z - made_mu) / made_sd
    log_detj = -tf.reduce_sum(tf.log(made_sd), 1)
    return z, log_detj

def make_loss(pred, actual, log_var, mu, log_detj, z0, sigma=1.0):
    q0 = tf.contrib.distributions.Normal(mu=mu, sigma=tf.exp(tf.maximum(1e-5, log_var))).pdf(z0)
    ln_q0 = tf.reduce_sum(tf.log(q0), 1)
    rec_err = tf.reduce_mean(crossentropy(pred, actual))

    loss = -tf.reduce_mean(ln_q0 + rec_err - log_detj)
    return loss

def train_step(sess, input_data, train_op, loss_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    _, l = sess.run([train_op, loss_op], feed_dict={x_op: input_data, e_op: e_})
    return l

def reconstruct(sess, input_data, out_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    x_rec = sess.run([out_op], feed_dict={x_op: input_data, e_op: e_})
    return x_rec


def sample_latent(sess, input_data, z_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    zs = sess.run(z_op, feed_dict={x_op: input_data, e_op: e_})
    return zs

def show_reconstruction(actual, recon):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(actual.reshape(28, 28), cmap='gray')
    axs[1].imshow(recon.reshape(28, 28), cmap='gray')
    axs[0].set_title('actual')
    axs[1].set_title('reconstructed')
    plt.show()


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data

    data = input_data.read_data_sets('data')
    data_dim = data.train.images.shape[1]
    enc_h = 128
    enc_z = 64
    dec_h = 128
    max_iters = 10000
    batch_size = 100
    learning_rate = 0.0005
    k = 5

    x, e = inputs(data_dim, enc_z)
    mu, log_var, z0 = encoder(x, e, data_dim, enc_h, enc_z, k)

    z_k, log_detj = inverse_autoregressive_flow(z0, enc_z, enc_h)

    out_op, out_mu, out_log_var = decoder(z_k, data_dim, dec_h, enc_z)

    loss_op = make_loss(out_op, x, log_var, mu, log_detj, z0)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    x_test, _ = data.test.next_batch(1)
    recons = []

    for i in xrange(max_iters):
        x_, y_ = data.train.next_batch(batch_size)
        l = train_step(sess, x_, train_op, loss_op, x, e, enc_z)
        if i % 1000 == 0:
            print('iter: %d\tavg. loss: %.2f' % (i, l/batch_size))
            recons.append(reconstruct(sess, x_test, out_op, x, e, enc_z)[0])

    for r in recons:
        show_reconstruction(x_test[0], r)

    sess.close()
