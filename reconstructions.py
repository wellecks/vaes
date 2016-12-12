import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def reconstruct(sess, input_data, out_op, x):
    x_rec = sess.run([out_op], feed_dict={x: input_data})
    return x_rec

def show_reconstruction(actual, recons, image_width):
    fig, axs = plt.subplots(1, len(recons) + 1)
    axs[0].set_title('actual')
    axs[0].imshow(actual.reshape(image_width, image_width), cmap='gray')
    for i, recon in enumerate(recons):
        axs[i+1].imshow(recon[0].reshape(image_width, image_width), cmap='gray')
        axs[i+1].set_title('reconstructed, iteration {}'.format(recon[1]))
    plt.show()

def vec2im(x, batch_size, image_width):
    shape = (batch_size, image_width, image_width, 1)
    return tf.reshape(x, shape) # reshape to 4D

def plot_images_together(images, image_width=28):
    """ Plot a single image containing all six MNIST images, one after
    the other.  Note that we crop the sides of the images so that they
    appear reasonably close together."""
    fig = plt.figure()
    image = np.concatenate(images, axis=1)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = 'gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()
