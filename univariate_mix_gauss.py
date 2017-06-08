# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import tensorflow as tf
from print_util import sfill, sfloat, sarray

# NUM_COMPONENTS = 2
# TRUE_PROBS = np.array([0.6, 0.4])
# TRUE_MU = np.array([-1.5, 1.5])
# TRUE_SIGMA = np.array([1.50, 0.50])
# SAMPLE_SIZE = 10000

NUM_COMPONENTS = 3
TRUE_PROBS = np.array([0.5, 0.3, 0.2])
TRUE_MU = np.array([-1.5, 0.0, 1.5])
TRUE_SIGMA = np.array([0.5, 0.4, 0.3])
SAMPLE_SIZE = 10000

if TRUE_PROBS.sum() != 1.0:
    raise Exception('Component weights should sum to 1.0')

INIT_LOGIT_PARAMS = {'mean': 0.0, 'stddev': 0.1}
INIT_MU_PARAMS = {'mean': 0.0, 'stddev': 0.1}
INIT_PHI_PARAMS = {'mean': 1.0, 'stddev': 0.1}
LEARNING_RATE = 0.001
MAX_ITER = 10000
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-8, 1e-8, 1e-8
RANDOM_SEED = 0

MAX_CHARS = 15

# generate sample
np.random.seed(0)
z_obs = np.random.choice(range(NUM_COMPONENTS),
                         size=SAMPLE_SIZE,
                         p=TRUE_PROBS)
x_obs = np.random.normal(loc=TRUE_MU[z_obs],
                         scale=TRUE_SIGMA[z_obs],
                         size=SAMPLE_SIZE)

# plot
# import matplotlib.pyplot as plt
#
# plt.hist([x_obs[z_obs == i] for i in range(NUM_COMPONENTS)],
#          bins=100, stacked=True, alpha=0.5, normed=True,
#          label=['component {}'.format(i + 1) for i in range(NUM_COMPONENTS)])
# plt.legend(loc='upper left')
# plt.show()

# center and scale the data
CENTER = x_obs.mean()
SCALE = x_obs.std()
x_obs = (x_obs - CENTER) / SCALE

# tensor for data
x = tf.placeholder(dtype=tf.float32)

# tensors representing parameters and variables
logit = tf.Variable(initial_value=tf.random_normal(shape=[NUM_COMPONENTS],
                                                   seed=RANDOM_SEED,
                                                   **INIT_LOGIT_PARAMS),
                    dtype=tf.float32)
p = tf.nn.softmax(logits=logit)
mu = tf.Variable(initial_value=tf.random_normal(shape=[NUM_COMPONENTS],
                                                seed=RANDOM_SEED,
                                                **INIT_MU_PARAMS),
                 dtype=tf.float32)
phi = tf.Variable(initial_value=tf.random_normal(shape=[NUM_COMPONENTS],
                                                 seed=RANDOM_SEED,
                                                 **INIT_PHI_PARAMS),
                  dtype=tf.float32)
sigma = tf.square(phi)

# loss function
categorical_dist = tf.contrib.distributions.Categorical(probs=p)
gaussian_dists = []
for i in range(NUM_COMPONENTS):
    gaussian_dists.append(tf.contrib.distributions.Normal(loc=mu[i],
                                                          scale=sigma[i]))
mixture_dist = tf.contrib.distributions.Mixture(cat=categorical_dist,
                                                components=gaussian_dists)
log_prob = mixture_dist.log_prob(value=x)
neg_log_likelihood = -1.0 * tf.reduce_sum(log_prob)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=neg_log_likelihood)

# gradient
grad = tf.gradients(neg_log_likelihood, [logit, mu, phi])

with tf.Session() as sess:
    sess.run(fetches=tf.global_variables_initializer())

    i = 1
    obs_logit, obs_p, obs_mu, obs_phi, obs_sigma = sess.run(
        fetches=[[logit], [p], [mu], [phi], [sigma]])
    obs_loss = sess.run(fetches=[neg_log_likelihood], feed_dict={x: x_obs})
    obs_grad = sess.run(fetches=[grad], feed_dict={x: x_obs})
    print(' {} | {} | {} | {} | {} | {}'
          .format(sfill('iter', len(str(MAX_ITER)), '>'),
                  sfill('p', MAX_CHARS + 2 * NUM_COMPONENTS, '^'),
                  sfill('mu', MAX_CHARS + 2 * NUM_COMPONENTS, '^'),
                  sfill('sigma', MAX_CHARS + 2 * NUM_COMPONENTS, '^'),
                  sfill('loss', MAX_CHARS, '^'),
                  sfill('grad', MAX_CHARS, '^')))

    while True:
        # gradient step
        sess.run(fetches=train_op, feed_dict={x: x_obs})

        # update parameters
        new_logit, new_p, new_mu, new_phi, new_sigma = sess.run(
            fetches=[logit, p, mu, phi, sigma])
        diff_norm = np.linalg.norm(np.subtract(
            [param for param_list in [new_logit, new_mu, new_phi]
             for param in param_list],
            [param for param_list in [obs_logit[-1], obs_mu[-1], obs_phi[-1]]
             for param in param_list]
        ))

        # update loss
        new_loss = sess.run(fetches=neg_log_likelihood, feed_dict={x: x_obs})
        loss_diff = np.abs(new_loss - obs_loss[-1])

        # update gradient
        new_grad = sess.run(fetches=grad, feed_dict={x: x_obs})
        grad_norm = np.linalg.norm(new_grad)

        obs_logit.append(new_logit)
        obs_p.append(new_p)
        obs_mu.append(new_mu)
        obs_phi.append(new_phi)
        obs_sigma.append(new_sigma)
        obs_loss.append(new_loss)
        obs_grad.append(new_grad)

        if (i - 1) % 100 == 0:
            print(' {} | {} | {} | {} | {} | {}'
                  .format(sfill(i, len(str(MAX_ITER))),
                          sarray(new_p, MAX_CHARS),
                          sarray(new_mu, MAX_CHARS),
                          sarray(new_sigma, MAX_CHARS),
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
print('Fitted MLE:')
for j in range(NUM_COMPONENTS):
    print('Component {}: [p={:.4f}, mu={:.4f}, sigma={:.4f}]'
          .format(j + 1, obs_p[-1][j],
                  SCALE * obs_mu[-1][j] + CENTER,
                  SCALE * obs_sigma[-1][j]))

print('True Values:')
for j in range(NUM_COMPONENTS):
    print('Component {}: [p={:.4f}, mu={:.4f}, sigma={:.4f}]'
          .format(j + 1, TRUE_PROBS[j], TRUE_MU[j], TRUE_SIGMA[j]))
