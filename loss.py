
import tensorflow as tf

def cross_entropy(obs, actual, offset=1e-7):
    """Binary cross-entropy, per training example"""
    # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
    with tf.name_scope("cross_entropy"):
        # bound by clipping to avoid nan
        obs_ = tf.clip_by_value(obs, offset, 1 - offset)
        return -tf.reduce_sum(actual * tf.log(obs_) +
                              (1 - actual) * tf.log(1 - obs_), 1)

def analytic_kl_divergence(mu, log_std):
    '''Compute the batched KL divergence between a diagonal Gaussian and a standard Gaussian'''
    kl = -(0.5*tf.reduce_sum(1.0 + 2 * log_std - tf.square(mu) - tf.exp(2 * log_std), 1))
    return kl

def gaussian_log_pdf(mu, log_std, z):
    '''Compute the log probability density of point z under Gaussian with mean mu and
    diagonal standard deviation exp(log_std)
    '''
    return tf.contrib.distributions.MultivariateNormalDiag(
                mu=mu, diag_stdev=tf.maximum(tf.exp(log_std), 1e-15)).log_pdf(z)

def elbo_loss(pred, actual, var_reg=1, kl_weighting=None, **kwargs):
    mu = kwargs['mu']
    log_std = kwargs['log_std']
    if 'sum_log_detj' not in kwargs:
        kl = tf.reduce_mean(analytic_kl_divergence(mu, log_std))
        rec_err = tf.reduce_mean(cross_entropy(pred, actual))

    else:
        sum_log_detj, z0, zk  = kwargs['sum_log_detj'], kwargs['z0'], kwargs['zk']
        sum_log_detj = tf.reduce_mean(sum_log_detj)
        log_q0_z0 = tf.reduce_mean(gaussian_log_pdf(mu, log_std, z0))
        log_qk_zk = log_q0_z0 - sum_log_detj
        log_p_zk = tf.reduce_mean(gaussian_log_pdf(tf.zeros_like(mu), tf.ones_like(mu), zk))
        log_p_x_given_zk = -tf.reduce_mean(cross_entropy(pred, actual))
        rec_err = -log_p_x_given_zk
        kl = log_qk_zk - log_p_zk
        tf.scalar_summary('Sum of log det Jacobians', sum_log_detj)
        tf.scalar_summary('Log q0(z0)', log_q0_z0)
        tf.scalar_summary('Log qk(zk)', log_qk_zk)
        tf.scalar_summary('Log p(zk)', log_p_zk)

    if kl_weighting is not None:
        loss =  kl_weighting * kl + rec_err
    else: loss = kl + rec_err

    tf.scalar_summary('Reconstruction error', rec_err)
    tf.scalar_summary('KL divergence', kl)
    tf.scalar_summary('ELBO', loss)

    return loss
