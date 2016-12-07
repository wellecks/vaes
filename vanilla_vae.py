"""Variational Auto-encoder

References
----------
https://arxiv.org/pdf/1312.6114v10.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import datetime
import inspect
import os
from tensorflow.examples.tutorials.mnist import input_data
from models import *
from reconstructions import *

def elbo_loss(pred, actual, var_reg=1, **kwargs):
    mu = kwargs['mu']
    log_std = kwargs['log_std']
    if 'sum_log_detj' not in kwargs:
        kl = -tf.reduce_mean(0.5*tf.reduce_sum(1.0 + 2 * log_std - tf.square(mu) - tf.exp(2 * log_std), 1))
        #rec_err = tf.reduce_mean(0.5*(tf.nn.l2_loss(actual - pred)/var_reg))
        #rec_err = -tf.reduce_mean(tf.contrib.distributions.MultivariateNormalDiag(
                    #mu=pred, diag_stdev=tf.ones_like(pred)).log_pdf(actual))
        rec_err = tf.reduce_mean(crossentropy(pred, actual))
        loss = kl + rec_err
        tf.scalar_summary('KL divergence', kl)
    else:
        #kl = -tf.reduce_mean(0.5*tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1))
        sum_log_detj, z0, zk  = kwargs['sum_log_detj'], kwargs['z0'], kwargs['zk']
        sum_log_detj = tf.reduce_mean(sum_log_detj)
        log_q0_z0 = tf.reduce_mean(tf.contrib.distributions.MultivariateNormalDiag(
                    mu=mu, diag_stdev=tf.maximum(tf.exp(log_std), 1e-15)).log_pdf(z0))
        log_qk_zk = log_q0_z0 - sum_log_detj
        log_p_zk = tf.reduce_mean(tf.contrib.distributions.MultivariateNormalDiag(
                    mu=tf.zeros_like(mu), diag_stdev=tf.ones_like(mu)).log_pdf(zk))
        #log_p_x_given_zk = tf.reduce_mean(tf.contrib.distributions.MultivariateNormalDiag(
        #            mu=pred, diag_stdev=tf.ones_like(pred)).log_pdf(actual))
        log_p_x_given_zk = -tf.reduce_mean(crossentropy(pred, actual))
        rec_err = log_p_x_given_zk
        #loss = tf.reduce_mean(kl + rec_err - log_detj)
        loss = log_qk_zk - log_p_zk  - log_p_x_given_zk
        tf.scalar_summary('Sum of log det Jacobians', sum_log_detj)
        tf.scalar_summary('Log q0(z0)', log_q0_z0)
        tf.scalar_summary('Log qk(zk)', log_qk_zk)
        tf.scalar_summary('Log p(zk)', log_p_zk)
        tf.scalar_summary('Log p(x|zk)', log_p_x_given_zk)

    tf.scalar_summary('ELBO', loss)

    tf.scalar_summary('Reconstruction error', rec_err)
    return loss

def train(
        image_width,
        dim_x,
        dim_z,
        encoder,
        decoder,
        learning_rate=0.0001,
        optimizer=tf.train.AdamOptimizer,
        loss=elbo_loss,
        batch_size=100,
        results_dir='results',
        max_steps=20000,
        data=input_data.read_data_sets('data'),

        **kwargs
        ):
    global_step = tf.Variable(0, trainable=False) # for checkpoint saving
    dt = datetime.datetime.now()
    results_dir += '/{:02d}-{:02d}-{:02d}_{}'.format(dt.hour, dt.minute, dt.second, dt.date())
    os.mkdir(results_dir)
    # Get all the settings and save them.
    with open(results_dir + '/settings.txt', 'w') as f:
        args = inspect.getargspec(train).args
        settings = [locals()[arg] for arg in args]
        for s, arg in zip(settings, args):
            f.write('{}: {}\n'.format(arg, s))

    # Build computation graph and operations
    x = tf.placeholder(tf.float32, [None, dim_x], 'x')
    e = tf.random_normal(shape=(batch_size, dim_z))
    #z_params, z = encoder(x, e, dim_x, enc_dims, dim_z)
    z_params, z = encoder(x, e)
    #z_params, z = encoder(x, e, dim_z, width=image_width, **kwargs) #CONVNET
    #x_pred = decoder(z, dim_x, dec_dims, dim_z)
    x_pred = decoder(z)
    loss_op = loss(x_pred, x, **z_params)
    #out_op = tf.sigmoid(x_pred)
    out_op = x_pred
    train_op = optimizer(learning_rate).minimize(loss_op, global_step=global_step)

    # Make summaries
    rec_summary = tf.image_summary("rec", vec2im(out_op, batch_size, image_width), max_images=10)
    summary_op = tf.merge_all_summaries()

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Create a session
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter(results_dir, sess.graph)
    x_test, _ = data.test.next_batch(1)
    recons = []

    for step in xrange(max_steps):
        start_time = time.time()
        x_, y_ = data.train.next_batch(batch_size)
        feed_dict={x: x_}
        _, l = sess.run([train_op, loss_op], feed_dict)

        duration = time.time() - start_time

        if step % 100 == 0:
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

        if step % 1000 == 0:
            print('iter: {:d}\tloss: {:.2f} ({:.1f} examples/sec)'.format(step, l, batch_size/duration))
            recons.append([[reconstruct(sess, x_test, out_op, x)[0][0], step]])

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_steps:
            checkpoint_path = os.path.join(results_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

    for r in recons:
        show_reconstruction(x_test[0], r, image_width)

    sess.close()

if __name__ == '__main__':

    '''
    This is where we put the training settings
    '''

    ### VANILLA VAE
    #dim_x, dim_z, enc_dims = 784, 96, [128]
    #encoder = basic_encoder(dim_x, enc_dims, dim_z)

    ### NORMALIZING FLOW
    dim_x, dim_z, enc_dims = 784, 96, [128]
    flow = 2
    encoder = nf_encoder(dim_x, enc_dims, dim_z, flow)

    dec_dims = [128]
    decoder = basic_decoder(dim_x, dec_dims, dim_z)

    train(
    image_width=28,
    dim_x=dim_x,
    dim_z=dim_z,
    encoder=encoder,
    decoder=decoder,

    learning_rate=0.0001,
    optimizer=tf.train.AdamOptimizer,
    loss=elbo_loss,
    batch_size=100,

    results_dir='results',
    max_steps=20000,
    #encoder=nf_encoder(1),

    #encoder=iaf_encoder(1),
    #encoder=conv_encoder(layer_dict, in_shape),
        )
