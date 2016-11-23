"""Variational Auto-encoder, class-based.

References
----------
https://arxiv.org/pdf/1312.6114v10.pdf
"""

import numpy as np
import tensorflow as tf

class VAE(object):
    def __init__(self, data_dim, z_dim, enc_h=128, dec_h=128, lr=0.01):
        self.initializer = tf.contrib.layers.xavier_initializer
        self.z_dim = z_dim
        self.x, self.e = self.inputs(data_dim, z_dim)
        self.mu, self.log_var, self.z = self.encoder(self.x, self.e, data_dim, enc_h, z_dim)
        self.out, self.out_mu, self.out_log_var = self.decoder(self.z, data_dim, dec_h, z_dim)
        self.loss_op = self.make_loss(self.out_mu, self.x, self.log_var, self.mu, self.out_log_var)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss_op)

    def inputs(self, D, Z):
        x = tf.placeholder(tf.float32, [None, D], 'x')
        e = tf.placeholder(tf.float32, [None, Z], 'e')
        return x, e

    def encoder(self, x, e, D, H, Z):
        with tf.variable_scope('encoder'):
            w_h = tf.get_variable('w_h', [D, H], initializer=self.initializer())
            b_h = tf.get_variable('b_h', [H], initializer=self.initializer())
            w_mu = tf.get_variable('w_mu', [H, Z], initializer=self.initializer())
            b_mu = tf.get_variable('b_mu', [Z], initializer=self.initializer())
            w_v = tf.get_variable('w_v', [H, Z], initializer=self.initializer())
            b_v = tf.get_variable('b_v', [Z], initializer=self.initializer())

            h = tf.nn.tanh(tf.matmul(x, w_h) + b_h)
            mu = tf.matmul(h, w_mu) + b_mu
            log_var = tf.matmul(h, w_v) + b_v
            z = mu + tf.sqrt(tf.exp(log_var)) * e
        return mu, log_var, z

    def decoder(self, z, D, H, Z):
        with tf.variable_scope('decoder'):
            w_h = tf.get_variable('w_h', [Z, H], initializer=self.initializer())
            b_h = tf.get_variable('b_h', [H], initializer=self.initializer())
            w_mu = tf.get_variable('w_mu', [H, D], initializer=self.initializer())
            b_mu = tf.get_variable('b_mu', [D], initializer=self.initializer())
            w_v = tf.get_variable('w_v', [H, 1], initializer=self.initializer())
            b_v = tf.get_variable('b_v', [1], initializer=self.initializer())

            h = tf.nn.tanh(tf.matmul(z, w_h) + b_h)
            out_mu = tf.matmul(h, w_mu) + b_mu
            out_log_var = tf.matmul(h, w_v) + b_v
            # NOTE(wellecks) Enforce 0, 1 (MNIST-specific)
            out = tf.sigmoid(out_mu)
        return out, out_mu, out_log_var

    def make_loss(self, pred, actual, log_var, mu, out_log_var):
        kl = 0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1)
        rec_err = -0.5 * (tf.nn.l2_loss(actual - pred))
        loss = -tf.reduce_mean(kl + rec_err)
        return loss

    def train_step(self, sess, input_data):
        e_ = np.random.normal(size=(input_data.shape[0], self.z_dim))
        _, loss = sess.run([self.train_op, self.loss_op], {self.x: input_data, self.e: e_})
        return loss

    def reconstruct(self, sess, input_data):
        e_ = np.random.normal(size=(input_data.shape[0], self.z_dim))
        x_rec = sess.run([self.out], {self.x: input_data, self.e: e_})
        return x_rec

    def sample_latent(self, sess, input_data):
        e_ = np.random.normal(size=(input_data.shape[0], self.z_dim))
        zs = sess.run(self.z, feed_dict={self.x: input_data, self.e: e_})
        return zs
