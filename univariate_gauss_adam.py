# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import tensorflow as tf
from util.sprint import sfill, sfloat, shess
from util.plot import plot_canonical_gauss

TRUE_MU = 10.0
TRUE_SIGMA = 5.0
SAMPLE_SIZE = 100

INIT_MU_PARAMS = {'loc': 0.0, 'scale': 0.1}
INIT_PHI_PARAMS = {'loc': 1.0, 'scale': 0.1}
LEARNING_RATE = 0.001
MAX_ITER = 10000
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-8, 1e-8, 1e-8
RANDOM_SEED = 0

MAX_CHARS = 15

# generate sample
np.random.seed(0)
x_obs = np.random.normal(loc=TRUE_MU, scale=TRUE_SIGMA, size=SAMPLE_SIZE)

# center and scale the data
CENTER = x_obs.min()
SCALE = x_obs.max() - x_obs.min()
x_obs = (x_obs - CENTER) / SCALE

# tensor for data
x = tf.placeholder(dtype=tf.float32)

# tensors for parameters
np.random.seed(RANDOM_SEED)
mu = tf.Variable(initial_value=np.random.normal(**INIT_MU_PARAMS),
                 dtype=tf.float32)
phi = tf.Variable(initial_value=np.random.normal(**INIT_PHI_PARAMS),
                  dtype=tf.float32)
sigma = tf.square(phi)

# loss function
gaussian_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
log_prob = gaussian_dist.log_prob(value=x)
neg_log_likelihood = -1.0 * tf.reduce_sum(log_prob)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=neg_log_likelihood)

# gradient
grad = tf.gradients(neg_log_likelihood, [mu, phi])

# hessian
hess = tf.stack(values=[tf.gradients(grad[0], [mu, phi]),
                        tf.gradients(grad[1], [mu, phi])], axis=0)

with tf.Session() as sess:
    # initialize
    sess.run(fetches=tf.global_variables_initializer())

    i = 1
    obs_mu, obs_phi, obs_sigma = sess.run(fetches=[[mu], [phi], [sigma]])
    obs_loss = sess.run(fetches=[neg_log_likelihood], feed_dict={x: x_obs})
    obs_grad = sess.run(fetches=[grad], feed_dict={x: x_obs})
    obs_hess = sess.run(fetches=[hess], feed_dict={x: x_obs})
    print(' {} | {} | {} | {} | {}'
          .format(sfill('iter', len(str(MAX_ITER)), '>'),
                  sfill('mu', MAX_CHARS, '^'),
                  sfill('sigma', MAX_CHARS, '^'),
                  sfill('loss', MAX_CHARS, '^'),
                  sfill('grad', MAX_CHARS, '^')))

    while True:
        # gradient step
        sess.run(fetches=train_op, feed_dict={x: x_obs})

        # update parameters
        new_mu, new_phi, new_sigma = sess.run(fetches=[mu, phi, sigma])
        diff_norm = np.linalg.norm(np.subtract([new_mu, new_phi],
                                               [obs_mu[-1], obs_phi[-1]]))

        # update loss
        new_loss = sess.run(fetches=neg_log_likelihood, feed_dict={x: x_obs})
        loss_diff = np.abs(new_loss - obs_loss[-1])

        # update gradient
        new_grad = sess.run(fetches=grad, feed_dict={x: x_obs})
        grad_norm = np.linalg.norm(new_grad)

        # update hessian
        new_hess = sess.run(fetches=hess, feed_dict={x: x_obs})

        obs_mu.append(new_mu)
        obs_phi.append(new_phi)
        obs_sigma.append(new_sigma)
        obs_loss.append(new_loss)
        obs_grad.append(new_grad)
        obs_hess.append(new_hess)

        if (i - 1) % 100 == 0:
            print(' {} | {} | {} | {} | {}'
                  .format(sfill(i, len(str(MAX_ITER))),
                          sfloat(new_mu, MAX_CHARS),
                          sfloat(new_sigma, MAX_CHARS),
                          sfloat(new_loss, MAX_CHARS),
                          sfloat(grad_norm, MAX_CHARS)))

        if diff_norm < TOL_PARAM:
            print('Parameter convergence in {} iterations!'.format(i))
            break

        if loss_diff < TOL_LOSS:
            print('Loss function convergence in {} iterations!'.format(i))
            break

        if grad_norm < TOL_GRAD:
            print('Gradient convergence in {} iterations!'.format(i))
            break

        if i >= MAX_ITER:
            print('Max number of iterations reached without convergence.')
            break

        i += 1

# print results
print('Fitted MLE: [{:.4f}, {:.4f}]'.format(obs_mu[-1], obs_sigma[-1]))
print('Target MLE: [{:.4f}, {:.4f}]'.format(x_obs.mean(), x_obs.std()))

# check hessians for positive definite
print('First {}'.format(shess(obs_hess[0], MAX_CHARS)))
print('Final {}'.format(shess(obs_hess[-1], MAX_CHARS)))
print('All Hessians are PD: {}'
      .format(np.all([np.all(np.linalg.eigvals(h) > 0) for h in obs_hess])))

# visualize results
plot_canonical_gauss(x_obs, obs_mu, obs_sigma, obs_loss,
                     title='canonical params, adam, alpha = {}'
                     .format(LEARNING_RATE))
