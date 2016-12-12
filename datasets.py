from collections import namedtuple
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

DATASETS_DIR = 'data'
DATASET_TYPES = ['train', 'valid', 'test']

def create_binarized_mnist(which):
    with open(DATASETS_DIR + '/binarized_mnist_{}.amat'.format(which)) as f:
        data = [l.strip().split(' ') for l in f.readlines()]
        data = np.array(data).astype(int)
        np.save(DATASETS_DIR + '/binarized_mnist_{}.npy'.format(which), data)
    return data

def binarized_mnist():
    dataset = {which: UnlabelledDataSet(np.load(DATASETS_DIR + '/binarized_mnist_{}.npy'.format(which)))for which in DATASET_TYPES}
    return dataset

class UnlabelledDataSet(object):

  def __init__(self,
               images):
    self._num_examples = images.shape[0]
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end]
