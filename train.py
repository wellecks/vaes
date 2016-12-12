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
from loss import *
from datasets import binarized_mnist
import argparse

def train(
        image_width,
        dim_x,
        dim_z,
        encoder,
        decoder,
        dataset,
        learning_rate=0.0001,
        optimizer=tf.train.AdamOptimizer,
        loss=elbo_loss,
        batch_size=100,
        results_dir='results',
        max_epochs=10,
        n_view=10,
        **kwargs
        ):
    global_step = tf.Variable(0, trainable=False) # for checkpoint saving
    on_epoch = tf.Variable(0, trainable=False)
    dt = datetime.datetime.now()
    results_dir += '/{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    os.mkdir(results_dir)
    # Get all the settings and save them.
    with open(results_dir + '/settings.txt', 'w') as f:
        args = inspect.getargspec(train).args
        settings = [locals()[arg] for arg in args]
        for s, arg in zip(settings, args):
            setting = '{}: {}'.format(arg, s)
            f.write('{}\n'.format(setting))
            print(setting)
        settings = locals()[inspect.getargspec(train).keywords]
        for kw, val in settings.items():
            setting = '{}: {}'.format(kw, val)
            f.write('{}\n'.format(setting))
            print(setting)

    # Build computation graph and operations
    x = tf.placeholder(tf.float32, [None, dim_x], 'x')
    e = tf.placeholder(tf.float32, (None, dim_z), 'noise')
    z_params, z = encoder(x, e)
    x_pred = decoder(z)
    #kl_weighting = tf.Variable(1.0 - tf.exp(-on_epoch / kl_annealing_rate)) if kl_annealing_rate is not None else None
    kl_weighting = 1
    loss_op = loss(x_pred, x, kl_weighting=kl_weighting, **z_params)
    out_op = x_pred
    lr = tf.Variable(learning_rate)
    train_op = optimizer(lr).minimize(loss_op, global_step=global_step)

    # Make training and validation sets
    training_data, validation_data = dataset['train'], dataset['valid']
    n_train_batches, n_valid_batches = training_data.images.shape[0] / batch_size, validation_data.images.shape[0] / batch_size,
    print 'Loaded training and validation data'
    visualized, e_visualized = validation_data.images[:n_view], np.random.normal(0, 1, (n_view, dim_z))

    # Make summaries
    rec_summary = tf.image_summary("rec", vec2im(out_op, batch_size, image_width), max_images=10)
    validation_summary = tf.scalar_summary("validation loss", loss_op)
    summary_op = tf.merge_all_summaries()

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Create a session
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter(results_dir, sess.graph)
    samples_list = []
    batch_counter = 0
    best_validation_loss = 1e100
    for epoch in range(max_epochs):
        on_epoch += 1
        start_time = time.time()
        for _ in xrange(n_train_batches):
            batch_counter += 1
            x_ = training_data.next_batch(batch_size)
            e_ = np.random.normal(0, 1, (batch_size, dim_z))
            feed_dict={x: x_, e: e_}
            _, l = sess.run([train_op, loss_op], feed_dict)

            if batch_counter % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, batch_counter)

            # Save the model checkpoint periodically.
            if batch_counter % 1000 == 0 or epoch == max_epochs:
                checkpoint_path = os.path.join(results_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

        l_v = 0
        for _ in range(n_valid_batches):
            x_valid = validation_data.next_batch(batch_size)
            e_valid = np.random.normal(0, 1, (batch_size, dim_z))
            l_v_batched = sess.run(loss_op, feed_dict={x: x_valid, e: e_valid})
            l_v += l_v_batched
        l_v /= n_valid_batches

        duration = time.time() - start_time
        examples_per_sec = (n_valid_batches + n_train_batches) * batch_size * 1.0 / duration
        print('Epoch: {:d}\t Training loss: {:.2f}, Validation loss {:.2f} ({:.1f} examples/sec, {:.1f} sec/epoch)'.format(epoch, l, l_v, examples_per_sec, duration))

        if l_v > best_validation_loss:
            lr /= 2
            learning_rate /= 2
            print "Annealing learning rate to {}".format(learning_rate)
        else: best_validation_loss = l_v

        samples = sess.run([out_op], feed_dict={x: visualized, e: e_visualized})
        samples = np.reshape(samples, (n_view, image_width, image_width))
        samples_list.append(samples)
        #show_samples(samples, image_width)

    for samples in samples_list:
        together = np.hstack((np.reshape(visualized, (n_view,image_width, image_width)), samples > 0.5))
        plot_images_together(together)

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--basic', action='store_true')
    group.add_argument('--nf', action='store_true')
    group.add_argument('--iaf', action='store_true')

    parser.add_argument('--flow', type=int, default=2)
    args = parser.parse_args()

    ### TRAINING SETTINGS
    dim_x, dim_z, enc_dims, dec_dims = 784, 40, [300, 300], [300, 300]
    encoder_net = lambda x: nn(x, enc_dims, name='encoder', act=tf.nn.tanh)
    #encoder_net = lambda x: conv_net(x, layer_dict)
    decoder_net = lambda z: nn(z, dec_dims, name='decoder', act=tf.nn.tanh)
    flow = args.flow

    ### ENCODER
    if args.basic:
        encoder = basic_encoder(encoder_net, dim_z)
    if args.nf:
        encoder = nf_encoder(encoder_net, dim_z, flow)
    if args.iaf:
        encoder = iaf_encoder(encoder_net, dim_z, flow)

    ### DECODER
    decoder = basic_decoder(decoder_net, dim_x)

    ##############

    #kl_annealing_rate = 5.0

    ### ENCODER
    #encoder, model_type = basic_encoder(encoder_net, dim_z), 'Vanilla VAE'
    #encoder, model_type = nf_encoder(encoder_net, dim_z, flow), 'Normalizing Flow'
    #encoder, model_type = hf_encoder(encoder_net, dim_z, flow), 'Householder Flow'
    #encoder, model_type = iaf_encoder(encoder_net, dim_z, flow), 'Inverse Autoregressive Flow'


    ### DECODER
    decoder = basic_decoder(decoder_net, dim_x)


    extra_settings = {
    #'model_type':model_type,
    'flow length':flow,
    'encoder structure':enc_dims,
    'decoder structure':dec_dims,
    #'kl annealing rate':kl_annealing_rate
    }

    #######################################
    ## TRAINING
    #######################################
    train(
    image_width=28,
    dim_x=dim_x,
    dim_z=dim_z,
    encoder=encoder,
    decoder=decoder,
    dataset=binarized_mnist(),
    learning_rate=0.001,
    optimizer=tf.train.AdamOptimizer,
    loss=elbo_loss,
    batch_size=100,

    results_dir='results',
    max_epochs=100,
    **extra_settings
        )
