"""Functions for restoring variables"""

import os
import tensorflow as tf

def get_saved_variable_values(checkpoint_directory, metagraph_name):
    """Return mapping of variable names to numpy arrays"""
    # HACK: create a temporary session, and store the raw values of the variables
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(checkpoint_directory, metagraph_name))
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_directory))

    variables = {}
    for v in tf.trainable_variables():
        variables[v.name] = v.eval(session=sess)

    # Clear the loaded variables from the graph
    tf.reset_default_graph()
    sess.close()
    return variables

def set_variables(sess, variable_values):
    """Set trainable variables in `sess` to the values contained in `variable_values`."""
    current_variables = {v.name: v for v in tf.trainable_variables()}
    for name, v in variable_values.iteritems():
        if name in current_variables:
            sess.run(current_variables[name].assign(v))
    del variable_values
    print('Set pre-trained variables')
