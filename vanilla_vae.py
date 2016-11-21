"""Variational Auto-encoder

References
----------
https://arxiv.org/pdf/1312.6114v10.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def inputs(D, Z):
    x = tf.placeholder(tf.float32, [None, D], 'x')
    e = tf.placeholder(tf.float32, [None, Z], 'e')
    return x, e

def encoder(x, e, D, H, Z, initializer=tf.contrib.layers.xavier_initializer):
    with tf.variable_scope('encoder'):
        w_h = tf.get_variable('w_h', [D, H], initializer=initializer())
        b_h = tf.get_variable('b_h', [H], initializer=initializer())
        w_mu = tf.get_variable('w_mu', [H, Z], initializer=initializer())
        b_mu = tf.get_variable('b_mu', [Z], initializer=initializer())
        w_v = tf.get_variable('w_v', [H, Z], initializer=initializer())
        b_v = tf.get_variable('b_v', [Z], initializer=initializer())

        h = tf.nn.tanh(tf.matmul(x, w_h) + b_h)
        mu = tf.matmul(h, w_mu) + b_mu
        log_var = tf.matmul(h, w_v) + b_v
        z = mu + tf.sqrt(tf.exp(log_var))*e
    return mu, log_var, z

def decoder(z, D, H, Z, initializer=tf.contrib.layers.xavier_initializer):
    with tf.variable_scope('decoder'):
        w_h = tf.get_variable('w_h', [Z, H], initializer=initializer())
        b_h = tf.get_variable('b_h', [H], initializer=initializer())
        w_mu = tf.get_variable('w_mu', [H, D], initializer=initializer())
        b_mu = tf.get_variable('b_mu', [D], initializer=initializer())
        w_v = tf.get_variable('w_v', [H, 1], initializer=initializer())
        b_v = tf.get_variable('b_v', [1], initializer=initializer())

        h = tf.nn.tanh(tf.matmul(z, w_h) + b_h)
        out_mu = tf.matmul(h, w_mu) + b_mu
        out_log_var = tf.matmul(h, w_v) + b_v
        # NOTE(wellecks) Enforce 0, 1 (MNIST-specific)
        out = tf.sigmoid(out_mu)
    return out, out_mu, out_log_var

def make_loss(pred, actual, log_var, mu, out_log_var):
    kl = 0.5*tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1)
    # NOTE(wellecks) `sigmoid_cross_entropy_with_logits` performs better than this explicit loss.
    # rec_err = -0.5*(tf.nn.l2_loss(actual - pred))
    rec_err = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(pred, actual), 1)
    loss = -tf.reduce_mean(kl + rec_err)
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
    max_iters = 20000
    batch_size = 100
    learning_rate = 0.001

    x, e = inputs(data_dim, enc_z)
    mu, log_var, z = encoder(x, e, data_dim, enc_h, enc_z)
    out_op, out_mu, out_log_var = decoder(z, data_dim, dec_h, enc_z)
    loss_op = make_loss(out_mu, x, log_var, mu, out_log_var)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    x_test, _ = data.test.next_batch(1)
    recons = []

    for i in xrange(max_iters):
        x_, y_ = data.train.next_batch(batch_size)
        l = train_step(sess, x_, train_op, loss_op, x, e, enc_z)
        if i % 1000 == 0:
            print('iter: %d\tloss: %.2f' % (i, l))
            recons.append(reconstruct(sess, x_test, out_op, x, e, enc_z)[0])

    for r in recons:
        show_reconstruction(x_test[0], r)

    sess.close()
