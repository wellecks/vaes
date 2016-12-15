
import tensorflow as tf

def cross_entropy(obs, actual, offset=1e-7):
    """Binary cross-entropy, per training example"""
    with tf.name_scope("cross_entropy"):
        # bound by clipping to avoid nan
        obs_ = tf.clip_by_value(obs, offset, 1 - offset)
        return -tf.reduce_sum(actual * tf.log(obs_) +
                              (1 - actual) * tf.log(1 - obs_), 1)

def l2_loss(obs, actual):
    with tf.name_scope("l2"):
        return tf.nn.l2_loss(obs - actual)

def analytic_kl_divergence(mu, log_std):
    """Compute the batched KL divergence between a diagonal Gaussian and a standard Gaussian"""
    kl = -(0.5*tf.reduce_sum(1.0 + 2 * log_std - tf.square(mu) - tf.exp(2 * log_std), 1))
    return kl

def gaussian_log_pdf(mu, log_std, z):
    """Compute the log probability density of point z under Gaussian with mean mu and
    diagonal standard deviation exp(log_std)"""
    return tf.contrib.distributions.MultivariateNormalDiag(
                mu=mu, diag_stdev=tf.maximum(tf.exp(log_std), 1e-15)).log_pdf(z)

def elbo_loss(pred, actual, var_reg=1, kl_weighting=1, rec_err_fn=cross_entropy, **kwargs):
    monitor_functions = {}

    mu = kwargs['mu']
    log_std = kwargs['log_std']
    if 'sum_log_detj' not in kwargs:
        raw_kl = tf.reduce_mean(analytic_kl_divergence(mu, log_std), name='kl')
        rec_err = tf.reduce_mean(rec_err_fn(pred, actual), name='rec_err')
    else:
        sum_log_detj, z0, zk  = kwargs['sum_log_detj'], kwargs['z0'], kwargs['zk']
        sum_log_detj = tf.reduce_mean(sum_log_detj, name='sum_log_detj')
        log_q0_z0 = tf.reduce_mean(gaussian_log_pdf(mu, log_std, z0), name='log_q0_z0')
        log_qk_zk = log_q0_z0 - sum_log_detj

        log_p_zk = tf.reduce_mean(gaussian_log_pdf(tf.zeros_like(mu), tf.zeros_like(mu), zk), name='log_p_zk')
        log_p_x_given_zk = -tf.reduce_mean(rec_err_fn(pred, actual), name='log_p_x_given_zk')
        rec_err = -log_p_x_given_zk
        raw_kl = log_qk_zk - log_p_zk
        monitor_functions.update({
            'log_p_zk': log_p_zk,
            'log_qk_zk': log_qk_zk,
            'log_p_x_given_zk': log_p_x_given_zk,
            'log_q0_z0': log_q0_z0,
            'sum_log_detj': sum_log_detj
        })

    weighted_kl = kl_weighting * raw_kl
    unweighted_elbo = raw_kl + rec_err
    weighted_elbo = weighted_kl + rec_err

    train_loss = weighted_elbo
    valid_loss = unweighted_elbo

    monitor_functions.update({
        'weighted_kl': weighted_kl,
        'unweighted_elbo': unweighted_elbo,
        'weighted_elbo': weighted_elbo,
        'raw_kl': raw_kl,
        'rec_err': rec_err,
        'train_loss': train_loss,
        'valid_loss': valid_loss
    })
    return monitor_functions
