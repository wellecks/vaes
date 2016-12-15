"""Train various variational auto-encoder models.

References
----------
https://arxiv.org/pdf/1312.6114v10.pdf
"""

import argparse
import datetime
import inspect
import os
import time

import numpy as np
import tensorflow as tf
try:
    from tensorflow.python import control_flow_ops
except ImportError:
    from tensorflow.python.ops import control_flow_ops

import restore
from models import *
from reconstructions import *
from loss import *
from datasets import binarized_mnist

def train(
        image_width,
        dim_x,
        dim_z,
        encoder_type,
        decoder,
        dataset,
        learning_rate=0.0001,
        optimizer=tf.train.AdamOptimizer,
        loss=elbo_loss,
        batch_size=100,
        results_dir='results',
        max_epochs=10,
        n_view=10,
        results_file=None,
        bn=False,
        **kwargs
        ):
    saved_variables = kwargs.pop('saved_variables', None)
    anneal_lr = kwargs.pop('anneal_lr', False)
    learning_rate_temperature = kwargs.pop('learning_rate_temperature', None)
    global_step = tf.Variable(0, trainable=False) # for checkpoint saving
    on_epoch = tf.placeholder(tf.float32, name='on_epoch')
    dt = datetime.datetime.now()
    results_file = results_file if results_file is not None else '/{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    results_dir += results_file
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

    # Make the neural neural_networks
    is_training = tf.placeholder(tf.bool)
    if bn:
        encoder_net = lambda x: nn(x, enc_dims, name='encoder', act=tf.nn.tanh, is_training=is_training)
    else:
        encoder_net = lambda x: nn(x, enc_dims, name='encoder', act=tf.nn.tanh, is_training=None)
    encoder = encoder_type(encoder_net, dim_z, flow)

    # Build computation graph and operations
    x = tf.placeholder(tf.float32, [None, dim_x], 'x')
    x_w = tf.placeholder(tf.float32, [None, dim_x], 'x_w')
    e = tf.placeholder(tf.float32, (None, dim_z), 'noise')

    z_params, z = encoder(x_w, e)
    x_pred = decoder(z)
    kl_weighting = 1.0 - tf.exp(-on_epoch / kl_annealing_rate) if kl_annealing_rate is not None else 1
    monitor_functions = loss(x_pred, x, kl_weighting=kl_weighting, **z_params)
    monitor_functions_sorted = sorted(monitor_functions.iteritems(), key=lambda x: x[0])
    monitor_output_train = {name: [] for name in monitor_functions.iterkeys()}
    monitor_output_valid = {name: [] for name in monitor_functions.iterkeys()}
    monitor_function_names = [p[0] for p in monitor_functions_sorted]
    monitor_function_list = [p[1] for p in monitor_functions_sorted]

    train_loss, valid_loss = monitor_functions['train_loss'], monitor_functions['valid_loss']

    out_op = x_pred

    # Batch normalization stuff
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        train_loss = control_flow_ops.with_dependencies([updates], train_loss)

    # Optimizer with gradient clipping
    lr = tf.Variable(learning_rate)
    optimizer = optimizer(lr)
    gvs = optimizer.compute_gradients(train_loss)
    capped_gvs = [(tf.clip_by_norm(grad, 1), var) if grad is not None else (grad, var)
                  for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # Make training and validation sets
    training_data, validation_data = dataset['train'], dataset['valid']
    n_train_batches, n_valid_batches = training_data.images.shape[0] / batch_size, validation_data.images.shape[0] / batch_size,
    print 'Loaded training and validation data'
    visualized, e_visualized = validation_data.images[:n_view], np.random.normal(0, 1, (n_view, dim_z))

    # Make summaries
    rec_summary = tf.image_summary("rec", vec2im(out_op, batch_size, image_width), max_images=10)
    for fn_name, fn in monitor_functions.items():
        tf.scalar_summary(fn_name, fn)
    summary_op = tf.merge_all_summaries()

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Create a session
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # Use pre-trained weight values
    if saved_variables is not None:
        restore.set_variables(sess, saved_variables)

    summary_writer = tf.train.SummaryWriter(results_dir, sess.graph)
    samples_list = []
    batch_counter = 0
    best_validation_loss = 1e100
    number_of_validation_failures = 0
    feed_dict = {}
    validation_losses, training_losses = [], []
    for epoch in range(max_epochs):
        feed_dict[on_epoch] = epoch
        start_time = time.time()
        l_t = 0
        monitor_output_epoch = {name: 0 for name in monitor_function_names}
        for _ in xrange(n_train_batches):
            batch_counter += 1
            feed_dict[x], feed_dict[x_w] = training_data.next_batch(batch_size, whitened=False)
            feed_dict[e] = np.random.normal(0, 1, (batch_size, dim_z))
            feed_dict[is_training] = True
            output = sess.run([train_op, train_loss] + monitor_function_list, feed_dict=feed_dict)
            l, monitor_output_batch = output[1], output[2:]

            for name, out in zip(monitor_function_names, monitor_output_batch):
                monitor_output_epoch[name] += out

            if batch_counter % 100 == 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, batch_counter)

            # Save the model checkpoint periodically.
            if batch_counter % 1000 == 0 or epoch == max_epochs:
                checkpoint_path = os.path.join(results_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)
            l_t += l
        l_t /= n_train_batches

        for name in monitor_function_names:
            monitor_output_train[name].append(monitor_output_epoch[name] / n_train_batches)

        training_losses.append(l_t)

        # Validation loop
        l_v = 0
        monitor_output_epoch = {name: 0 for name in monitor_function_names}
        for _ in range(n_valid_batches):
            feed_dict[x], feed_dict[x_w] = validation_data.next_batch(batch_size, whitened=False)
            feed_dict[e] = np.random.normal(0, 1, (batch_size, dim_z))
            feed_dict[is_training] = False
            output = sess.run([valid_loss] + monitor_function_list, feed_dict=feed_dict)
            l_v_batched, monitor_output_batch = output[0], output[1:]
            for name, out in zip(monitor_function_names, monitor_output_batch):
                monitor_output_epoch[name] += out
            l_v += l_v_batched

        l_v /= n_valid_batches
        for name in monitor_function_names:
            monitor_output_valid[name].append(monitor_output_epoch[name] / n_valid_batches)

        validation_losses.append(l_v)
        duration = time.time() - start_time
        examples_per_sec = (n_valid_batches + n_train_batches) * batch_size * 1.0 / duration
        print('Epoch: {:d}\t Weighted training loss: {:.2f}, Validation loss {:.2f} ({:.1f} examples/sec, {:.1f} sec/epoch)'.format(epoch, l, l_v, examples_per_sec, duration))

        samples = sess.run([out_op], feed_dict={x: visualized, x_w: visualized, e: e_visualized, is_training: False})
        samples = np.reshape(samples, (n_view, image_width, image_width))
        samples_list.append(samples)
        # show_samples(samples, image_width)

        # Learning rate annealing
        lr = lr / (1.0 + epoch * 1.0 / learning_rate_temperature) if learning_rate_temperature is not None else lr

        if epoch % 100 == 0:
            np.save(results_dir + '/validation_losses_{}.npy'.format(epoch), validation_losses)
            np.save(results_dir + '/training_losses_{}.npy'.format(epoch), training_losses)
            np.save(results_dir + '/sample_visualizations_{}.npy'.format(epoch), np.array(samples_list))
            np.save(results_dir + '/real_visualizations_{}.npy'.format(epoch), np.reshape(visualized, (n_view,image_width, image_width)))
            for name in monitor_function_names:
                np.save(results_dir + '/{}_valid_{}.npy'.format(name, epoch), monitor_output_valid[name])
                np.save(results_dir + '/{}_train_{}.npy'.format(name, epoch), monitor_output_train[name])

    np.save(results_dir + '/validation_losses.npy', validation_losses)
    np.save(results_dir + '/training_losses.npy', training_losses)
    np.save(results_dir + '/sample_visualizations.npy', np.array(samples_list))
    np.save(results_dir + '/real_visualizations.npy', np.reshape(visualized, (n_view,image_width, image_width)))
    for name in monitor_function_names:
        np.save(results_dir + '/{}_valid.npy'.format(name), monitor_output_valid[name])
        np.save(results_dir + '/{}_train.npy'.format(name), monitor_output_train[name])

    visualize = False
    if visualize:
        for samples in samples_list:
            together = np.hstack((np.reshape(visualized, (n_view,image_width, image_width)), samples > 0.5))
            plot_images_together(together)

    sess.close()


def train_simple(
        dim_x,
        dim_z,
        encoder,
        decoder,
        training_dataset,
        validation_dataset=None,
        learning_rate=0.0001,
        optimizer=tf.train.AdamOptimizer,
        batch_size=100,
        max_epochs=10,
        **kwargs):
    print_every = kwargs.pop('print_every', 10)
    # Set random seeds
    seed = kwargs.pop('seed', 0)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    rec_err_fn = l2_loss if kwargs.pop('rec_err_type', '') == 'l2_loss' else cross_entropy
    anneal_lr = kwargs.pop('anneal_lr', False)

    # Build computation graph and operations
    x = tf.placeholder(tf.float32, [None, dim_x], 'x')
    e = tf.placeholder(tf.float32, (None, dim_z), 'noise')
    z_params, z = encoder(x, e)
    x_pred = decoder(z)

    kl_weighting = 1
    loss_op = elbo_loss(x_pred, x, kl_weighting=kl_weighting, rec_err_fn=rec_err_fn, **z_params)
    out_op = x_pred
    lr = tf.Variable(learning_rate)
    train_op = optimizer(lr).minimize(loss_op)

    # Make training and validation sets
    n_train_batches = max(training_dataset.num_examples / batch_size, 1)
    n_valid_batches = validation_dataset.num_examples / batch_size if validation_dataset is not None else 0

    # Create a session
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    batch_counter = 0
    best_validation_loss = 1e100
    for epoch in range(max_epochs):
        for _ in xrange(n_train_batches):
            batch_counter += 1

            x_ = training_dataset.next_batch(batch_size)
            e_ = np.random.normal(0, 1, (x_.shape[0], dim_z))
            feed_dict = {x: x_, e: e_}
            _, l = sess.run([train_op, loss_op], feed_dict)

        l_v = 0.0
        if validation_dataset is not None:
            for _ in range(n_valid_batches):
                x_valid = validation_dataset.next_batch(batch_size)
                e_valid = np.random.normal(0, 1, (batch_size, dim_z))
                l_v_batched = sess.run(loss_op, feed_dict={x: x_valid, e: e_valid})
                l_v += l_v_batched
            l_v /= n_valid_batches

            if l_v > best_validation_loss:
                if anneal_lr:
                    lr /= 2
                    learning_rate /= 2
                    print "Annealing learning rate to {}".format(learning_rate)
            else: best_validation_loss = l_v

        if (epoch + 1) % print_every == 0:
            print('Epoch: {:d}\t Training loss: {:.2f}'.format(epoch+1, l))

    ops = {
        'z': z,
        'out': out_op,
        'x': x,
        'e': e
    }
    return ops, sess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--basic', action='store_true')
    group.add_argument('--nf', action='store_true')
    group.add_argument('--iaf', action='store_true')
    group.add_argument('--hf', action='store_true')
    group.add_argument('--liaf', action='store_true')

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--anneal-lr', action='store_true')
    parser.add_argument('--flow', type=int, default=1)
    parser.add_argument('--lrt', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pretrained-metagraph', default=None)
    args = parser.parse_args()

    # Load pretrained variables
    if args.pretrained_metagraph is not None:
        s = args.pretrained_metagraph
        checkpoint_dir, metagraph_name = '/'.join(s.split('/')[:-1]), s.split('/')[-1]
        saved_variables = restore.get_saved_variable_values(checkpoint_dir, metagraph_name)
    else:
        saved_variables = None

    # Set random seeds
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Results file
    dt = datetime.datetime.now()
    results_file = '/{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)

    # TRAINING SETTINGS
    dim_x, dim_z, enc_dims, dec_dims = 784, 40, [300, 300], [300, 300]
    decoder_net = lambda z: nn(z, dec_dims, name='decoder', act=tf.nn.tanh)
    flow = args.flow
    bn = True

    # ENCODER
    if args.basic:
        encoder_type = basic_encoder
        results_file += '-basic'
    if args.nf:
        encoder_type = nf_encoder
        results_file += '-NF-{}'.format(flow)
    if args.iaf:
        encoder_type = iaf_encoder
        results_file += '-IAF-{}'.format(flow)
    if args.hf:
        encoder_type = hf_encoder
        results_file += '-HF-{}'.format(flow)
    if args.liaf:
        encoder_type = linear_iaf_encoder
        results_file += '-linIAF'

    if args.pretrained_metagraph is not None:
        results_file += '_pretrained'

    decoder = basic_decoder(decoder_net, dim_x)

    kl_annealing_rate = None
    extra_settings = {
        'flow': flow,
        'kl annealing rate': kl_annealing_rate,
        'anneal_lr': args.anneal_lr,
        'bn': bn,
        'enc_dims': enc_dims,
        'learning_rate_temperature': args.lrt
    }

    # TRAINING
    train(
        image_width=28,
        dim_x=dim_x,
        dim_z=dim_z,
        encoder_type=encoder_type,
        decoder=decoder,
        dataset=binarized_mnist(),
        learning_rate=0.0002,
        optimizer=tf.train.AdamOptimizer,
        loss=elbo_loss,
        batch_size=100,
        results_dir='results',
        results_file=results_file,
        max_epochs=args.epochs,
        saved_variables=saved_variables,
        **extra_settings
    )
