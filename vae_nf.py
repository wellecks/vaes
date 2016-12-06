"""Variational Auto-encoder with Normalizing Flows

References
----------
https://arxiv.org/pdf/1505.05770v6.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

        # Weights for outputting normalizing flow parameters
        w_us = tf.get_variable('w_us', [H, K*Z])
        b_us = tf.get_variable('b_us', [K*Z])
        w_ws = tf.get_variable('w_ws', [H, K*Z])
        b_ws = tf.get_variable('b_ws', [K*Z])
        w_bs = tf.get_variable('w_bs', [H, K])
        b_bs = tf.get_variable('b_bs', [K])

        h = tf.nn.tanh(tf.matmul(x, w_h) + b_h)
        mu = tf.matmul(h, w_mu) + b_mu
        log_var = tf.matmul(h, w_v) + b_v
        z = mu + tf.sqrt(tf.exp(log_var))*e

        # Normalizing Flow parameters
        us = tf.matmul(h, w_us) + b_us
        ws = tf.matmul(h, w_ws) + b_ws
        bs = tf.matmul(h, w_bs) + b_bs

        lambd = (us, ws, bs)

    return mu, log_var, z, lambd

def dtanh(tensor):
    return 1.0 - tf.square(tf.tanh(tensor))

def norm_flow(z, lambd, K, Z):
    us, ws, bs = lambd

    log_detjs = []
    for k in range(K):
        u, w, b = us[:, k*Z:(k+1)*Z], ws[:, k*Z:(k+1)*Z], bs[:, k]
        temp = tf.expand_dims(tf.nn.tanh(tf.reduce_sum(w*z, 1) + b), 1)
        temp = tf.tile(temp, [1, u.get_shape()[1].value])
        z = z + tf.mul(u, temp)

        # Eqn. (11) and (12)
        temp = tf.expand_dims(dtanh(tf.reduce_sum(w*z, 1) + b), 1)
        temp = tf.tile(temp, [1, w.get_shape()[1].value])
        log_detj = tf.abs(1. + tf.reduce_sum(tf.mul(u, temp*w), 1))
        log_detjs.append(log_detj)

    if K != 0:
        log_detj = tf.reduce_sum(log_detjs)
    else: log_detj = 0

    return z, log_detj


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

def make_loss(pred, actual, log_var, mu, log_detj, z0, sigma=1.0):
    #q0 = tf.contrib.distributions.Normal(mu=mu, sigma=tf.exp(log_var)).pdf(z0)
    #ln_q0 = tf.reduce_sum(tf.log(q0), 1)
    kl = -tf.reduce_mean(0.5*tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1))
    rec_err = 0.5*(tf.nn.l2_loss(actual - pred)) / sigma
    #loss = tf.reduce_mean(ln_q0 + rec_err - log_detj)
    loss = tf.reduce_mean(kl + rec_err - log_detj)
    return loss

def train_step(sess, input_data, train_op, loss_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    _, l = sess.run([train_op, loss_op], feed_dict={x_op: input_data, e_op: e_})
    return l

def reconstruct(sess, input_data, out_op, x_op, e_op, Z):
    e_ = np.random.normal(size=(input_data.shape[0], Z))
    x_rec = sess.run([out_op], feed_dict={x_op: input_data, e_op: e_})
    return x_rec

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
    batch_size = 128
    learning_rate = 0.001
    k = 0

    x, e = inputs(data_dim, enc_z)
    mu, log_var, z0, lambd = encoder(x, e, data_dim, enc_h, enc_z, k)

    z_k, log_detj = norm_flow(z0, lambd, k, enc_z)

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
