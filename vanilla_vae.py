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
    e = tf.placeholder(tf.float32, (None, dim_z), 'noise')
    z_params, z = encoder(x, e)
    x_pred = decoder(z)
    loss_op = loss(x_pred, x, **z_params)
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

    '''
    This is where we put the training settings
    '''

    layer_dict = {
        'wc1': [5, 5, 1, 32],
        'wc2': [5, 5, 32, 64],
        'wd1': [7*7*64, 128],
        'out': [128, 96]
    }

    ##############
    dim_x, dim_z, enc_dims, dec_dims = 784, 40, [300, 300], [300, 300]
    encoder_net = lambda x: nn(x, enc_dims, name='encoder', act=tf.nn.tanh)
    #encoder_net = lambda x: conv_net(x, layer_dict)
    decoder_net = lambda z: nn(z, dec_dims, name='decoder', act=tf.nn.tanh)
    flow = 1
    model_type = 'Vanilla VAE'
    model_type = 'Normalizing Flow'
    model_type = 'Inverse Autoregressive Flow'

    ### ENCODER
    #encoder = basic_encoder(encoder_net, dim_z)
    #encoder = nf_encoder(encoder_net, dim_z, flow)
    encoder = hf_encoder(encoder_net, dim_z, flow)
    #encoder = iaf_encoder(encoder_net, dim_z, flow)


    ### DECODER
    decoder = basic_decoder(decoder_net, dim_x)

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
        )
