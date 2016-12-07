"""Variational Auto-encoder

References
----------
https://arxiv.org/pdf/1312.6114v10.pdf
"""

import time
import datetime
import inspect
import os
from tensorflow.examples.tutorials.mnist import input_data
from models import *
from reconstructions import *
from loss import *

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
    z_params, z = encoder(x, e)
    x_pred = decoder(z)
    loss_op = loss(x_pred, x, **z_params)
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
    dim_x, dim_z, enc_dims, dec_dims = 784, 96, [128], [128]
    encoder_net = lambda x: nn(x, enc_dims, name='encoder')
    decoder_net = lambda z: nn(z, dec_dims, name='decoder')
    flow = 2

    ### ENCODER
    #encoder = basic_encoder(encoder_net, dim_z)
    encoder = nf_encoder(encoder_net, dim_z, flow)
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

    learning_rate=0.0001,
    optimizer=tf.train.AdamOptimizer,
    loss=elbo_loss,
    batch_size=100,

    results_dir='results',
    max_steps=20000,
        )
