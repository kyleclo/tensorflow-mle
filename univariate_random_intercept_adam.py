# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import tensorflow as tf
from util.sprint import sfill, sfloat, sarray

# random intercepts-only
NUM_SUBJECTS = 100
NUM_OBS_PER_SUBJECT = 10
# NUM_OBS_PER_SUBJECT = 100
TRUE_MU = 5.0
TRUE_EPSILON_SIGMA = 0.5
TRUE_GAMMA_SIGMA = 3.0

INIT_MU_PARAMS = {'mean': 0.0, 'stddev': 0.1}
INIT_PHI_PARAMS = {'mean': 1.0, 'stddev': 0.1}
LEARNING_RATE = 0.001
MAX_ITER = 10000
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-8, 1e-8, 1e-8
RANDOM_SEED = 0

MAX_CHARS = 10

# generate sample
np.random.seed(0)
gamma_obs = np.random.normal(loc=0.0,
                             scale=TRUE_GAMMA_SIGMA,
                             size=NUM_SUBJECTS)
epsilon_obs = np.random.normal(loc=0.0,
                               scale=TRUE_EPSILON_SIGMA,
                               size=NUM_SUBJECTS * NUM_OBS_PER_SUBJECT)
id_obs = np.concatenate([np.repeat(i, NUM_OBS_PER_SUBJECT)
                         for i in range(NUM_SUBJECTS)])
y_obs = TRUE_MU + gamma_obs[id_obs] + epsilon_obs

# center and scale the data
CENTER = y_obs.mean()
SCALE = y_obs.std()
y_obs = (y_obs - CENTER) / SCALE

# tensor for data
y = tf.placeholder(dtype=tf.float32)
id = tf.placeholder(dtype=tf.int32)

# tensors representing parameters and variables
mu = tf.Variable(initial_value=tf.random_normal(shape=[],
                                                seed=RANDOM_SEED,
                                                **INIT_MU_PARAMS),
                 dtype=tf.float32)
gamma = tf.Variable(initial_value=tf.random_normal(shape=[NUM_SUBJECTS],
                                                   seed=RANDOM_SEED,
                                                   **INIT_MU_PARAMS))
gamma_phi = tf.Variable(initial_value=tf.random_normal(shape=[],
                                                       seed=RANDOM_SEED,
                                                       **INIT_PHI_PARAMS),
                        dtype=tf.float32)
gamma_sigma = tf.square(gamma_phi)
epsilon_phi = tf.Variable(initial_value=tf.random_normal(shape=[],
                                                         seed=RANDOM_SEED,
                                                         **INIT_PHI_PARAMS),
                          dtype=tf.float32)
epsilon_sigma = tf.square(epsilon_phi)

# loss function
gamma_dist = tf.contrib.distributions.Normal(loc=0.0, scale=gamma_sigma)
gamma_log_prob = gamma_dist.log_prob(value=gamma)
x = tf.one_hot(indices=id, depth=NUM_SUBJECTS)
y_dist = tf.contrib.distributions.Normal(loc=mu,
                                         scale=epsilon_sigma)
y_log_prob = y_dist.log_prob(value=tf.expand_dims(y, 1) -
                                   tf.matmul(x, tf.expand_dims(gamma, 1)))
neg_log_likelihood = -1.0 * (tf.reduce_sum(y_log_prob) +
                             tf.reduce_sum(gamma_log_prob))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=neg_log_likelihood)

# gradient
grad_univar = tf.gradients(neg_log_likelihood, [mu, epsilon_phi, gamma_phi])
grad_gamma = tf.gradients(neg_log_likelihood, gamma)

with tf.Session() as sess:
    sess.run(fetches=tf.global_variables_initializer())

    i = 1
    obs_mu, obs_epsilon_phi, obs_epsilon_sigma, \
    obs_gamma, obs_gamma_phi, obs_gamma_sigma = \
        sess.run(fetches=[[mu], [epsilon_phi], [epsilon_sigma],
                          [gamma], [gamma_phi], [gamma_sigma]])
    obs_loss = sess.run(fetches=[neg_log_likelihood], feed_dict={y: y_obs,
                                                                 id: id_obs})
    obs_grad_univar = sess.run(fetches=[grad_univar], feed_dict={y: y_obs,
                                                                 id: id_obs})
    obs_grad_gamma = sess.run(fetches=[grad_gamma], feed_dict={y: y_obs,
                                                               id: id_obs})
    print(' {} | {} | {} | {} | {} | {} | {}'
          .format(sfill('iter', len(str(MAX_ITER)), '>'),
                  sfill('mu', MAX_CHARS, '^'),
                  sfill('e_sigma', MAX_CHARS, '^'),
                  sfill('|gamma|', MAX_CHARS, '^'),
                  sfill('g_sigma', MAX_CHARS, '^'),
                  sfill('loss', MAX_CHARS, '^'),
                  sfill('grad', MAX_CHARS, '^')))

    while True:
        # gradient step
        sess.run(fetches=train_op, feed_dict={y: y_obs, id: id_obs})

        # update parameters
        new_mu, new_epsilon_phi, new_epsilon_sigma, \
        new_gamma, new_gamma_phi, new_gamma_sigma = \
            sess.run(fetches=[mu, epsilon_phi, epsilon_sigma,
                              gamma, gamma_phi, gamma_sigma])
        diff_norm = np.linalg.norm(np.subtract(
            [param for param_list in [[new_mu], [new_epsilon_phi],
                                      new_gamma, [new_gamma_phi]]
             for param in param_list],
            [param for param_list in [[obs_mu[-1]], [obs_epsilon_phi[-1]],
                                      obs_gamma[-1], [obs_gamma_phi[-1]]]
             for param in param_list]
        ))

        # update loss
        new_loss = sess.run(fetches=neg_log_likelihood, feed_dict={y: y_obs,
                                                                   id: id_obs})
        loss_diff = np.abs(new_loss - obs_loss[-1])

        # update gradient
        new_grad_univar = sess.run(fetches=grad_univar, feed_dict={y: y_obs,
                                                                   id: id_obs})
        new_grad_gamma = sess.run(fetches=grad_gamma, feed_dict={y: y_obs,
                                                                 id: id_obs})
        grad_norm = np.sqrt(np.inner(new_grad_univar, new_grad_univar) + \
                            np.inner(new_grad_gamma, new_grad_gamma))

        obs_mu.append(new_mu)
        obs_epsilon_phi.append(new_epsilon_phi)
        obs_epsilon_sigma.append(new_epsilon_sigma)
        obs_gamma.append(new_gamma)
        obs_gamma_phi.append(new_gamma_phi)
        obs_gamma_sigma.append(new_gamma_sigma)
        obs_loss.append(new_loss)
        obs_grad_univar.append(new_grad_univar)
        obs_grad_gamma.append(new_grad_gamma)

        if (i - 1) % 100 == 0:
            print(' {} | {} | {} | {} | {} | {} | {}'
                  .format(sfill(i, len(str(MAX_ITER))),
                          sfloat(new_mu, MAX_CHARS),
                          sfloat(new_epsilon_sigma, MAX_CHARS),
                          sfloat(np.linalg.norm(new_gamma), MAX_CHARS),
                          sfloat(new_gamma_sigma, MAX_CHARS),
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
print('Fitted MLE: [{:.4f}, {:.4f}, {:.4f}]'
      .format(SCALE * obs_mu[-1] + CENTER,
              SCALE * obs_epsilon_sigma[-1],
              SCALE * obs_gamma_sigma[-1]))
print('True Values: [{:.4f}, {:.4f}, {:.4f}]'
      .format(TRUE_MU,
              TRUE_EPSILON_SIGMA,
              TRUE_GAMMA_SIGMA))

