import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util.sprint import sfill, sfloat

NUM_OBS = 100
NUM_CPTS = 3
TRUE_CPTS = np.array([20, 30, 75])

# from the changepoints, we can compute segments
TRUE_SEGMENT_LENGTHS = np.diff([0] + TRUE_CPTS.tolist() + [NUM_OBS])
TRUE_SEGMENT_LABELS = [j for j in range(NUM_CPTS + 1)
                       for _ in range(TRUE_SEGMENT_LENGTHS[j])]
TRUE_SIGMA = np.array([2, 5, 1, 3])

TRUE_INTERCEPT = 1
TRUE_SLOPE = 5
TRUE_DELTAS = np.array([-9, 3, 1])
TRUE_PARAMS = np.array([TRUE_INTERCEPT, TRUE_SLOPE] + TRUE_DELTAS.tolist())

INIT_BETA_PARAMS = {'mean': 0.0, 'stddev': 1.0}
SMOOTHING_PARAM = 1.0
LEARNING_RATE = 0.001
MAX_ITER = 10000
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-8, 1e-8, 1e-8
RANDOM_SEED = 0

MAX_CHARS = 15


def hinge(z):
    return np.max([0, z])


# generate time grid
t_grid = range(1, NUM_OBS + 1)


def evaluate_spline(t):
    """Evaluates spline with known knots at time t"""
    return TRUE_INTERCEPT + TRUE_SLOPE * t + \
           np.sum([delta * hinge(t - cpt)
                   for delta, cpt in zip(TRUE_DELTAS, TRUE_CPTS)])


# generate mean trend using spline formulation
np.random.seed(0)
mu_grid = [evaluate_spline(t) for t in t_grid]


def create_design_matrix(t_grid):
    """Creates design matrix for spline with known notes over grid of time"""
    columns = [np.repeat(1, NUM_OBS), t_grid]
    columns.extend([[hinge(t - cpt) for t in t_grid] for cpt in TRUE_CPTS])
    return np.stack(columns, axis=1)


# generate mean trend using design matrix formulation (equivalent to above)
np.random.seed(0)
mu_grid = np.matmul(create_design_matrix(t_grid), TRUE_PARAMS)

# generate noisy observations around mean trend
y_obs = np.array([np.random.normal(loc=mu_grid[t],
                                   scale=TRUE_SIGMA[TRUE_SEGMENT_LABELS[t]])
                  for t in range(NUM_OBS)]).reshape(-1, 1)

# center and scale the data
CENTER = y_obs.mean()
SCALE = y_obs.std()
y_obs = (y_obs - CENTER) / SCALE

# tensor for data
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
x = tf.placeholder(dtype=tf.float32, shape=[None, NUM_OBS])

# tensors representing parameters
beta = tf.Variable(initial_value=tf.random_normal(shape=[NUM_OBS, 1],
                                                  seed=RANDOM_SEED,
                                                  **INIT_BETA_PARAMS),
                   dtype=tf.float32)

# differnece matrix is constant
D = np.zeros(shape=[NUM_OBS - 2, NUM_OBS])
for index_row in range(NUM_OBS - 2):
    D[index_row, index_row] = 1.0
    D[index_row, index_row + 1] = -2.0
    D[index_row, index_row + 2] = 1.0
D = tf.constant(value=D, dtype=tf.float32)

# smoothing param
smoothing_param = tf.constant(SMOOTHING_PARAM, dtype=tf.float32)

# loss function
squared_error = tf.reduce_sum(tf.square(y - beta))
# squared_error = tf.reduce_sum(tf.square(y - tf.matmul(x, beta)))
regularizer = smoothing_param * tf.norm(tf.matmul(D, beta), ord=1)
fused_lasso_loss = squared_error + regularizer

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=fused_lasso_loss)

# gradient
grad_beta = tf.gradients(fused_lasso_loss, beta)

with tf.Session() as sess:
    sess.run(fetches=tf.global_variables_initializer())

    i = 1
    obs_beta = sess.run(fetches=[beta])
    obs_loss = sess.run(fetches=[fused_lasso_loss], feed_dict={y: y_obs})
    obs_grad_beta = sess.run(fetches=[grad_beta], feed_dict={y: y_obs})

    print(' {} | {} | {} | {}'
          .format(sfill('iter', len(str(MAX_ITER)), '>'),
                  sfill('|beta|', MAX_CHARS, '^'),
                  sfill('loss', MAX_CHARS, '^'),
                  sfill('grad', MAX_CHARS, '^')))

    while True:
        # gradient step
        sess.run(fetches=train_op, feed_dict={y: y_obs})

        # update parameters
        new_beta = sess.run(fetches=beta)
        diff_norm = np.linalg.norm(np.subtract(new_beta, obs_beta))

        # update loss
        new_loss = sess.run(fetches=fused_lasso_loss, feed_dict={y: y_obs})
        loss_diff = np.abs(new_loss - obs_loss[-1])

        # update gradient
        new_grad_beta = sess.run(fetches=grad_beta, feed_dict={y: y_obs})
        grad_norm = np.linalg.norm(new_grad_beta)

        obs_beta.append(new_beta)
        obs_loss.append(new_loss)
        obs_grad_beta.append(new_grad_beta)

        if (i - 1) % 100 == 0:
            print(' {} | {} | {} | {}'
                  .format(sfill(i, len(str(MAX_ITER))),
                          sfloat(np.linalg.norm(new_beta), MAX_CHARS),
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

# visualize results
plt.plot(t_grid, mu_grid, color='grey', label='mean')
plt.scatter(x=t_grid, y=SCALE * y_obs + CENTER, s=7, color='blue', marker='o',
            label='obs')
plt.plot(t_grid, SCALE * obs_beta[-1] + CENTER, color='red', label='est')
plt.legend(loc='upper right')
