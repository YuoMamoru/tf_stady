import os

import numpy as np
from scipy import stats
from sklearn.datasets import fetch_california_housing
import tensorflow as tf


housing = fetch_california_housing()
m, n = housing.data.shape
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)),
                                      stats.zscore(housing.data)]

n_epocks = 5000
learning_rate = 0.01
tf.random.set_random_seed(0)
calc_grad = True
base_dir = os.path.dirname(__file__)

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')
if calc_grad:
    # gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    gradients = tf.gradients(mse, [theta])[0]
    training_op = tf.assign(theta, theta - learning_rate * gradients)
else:  # Use optimizer of Tensorflow
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
    #                                        momentum=0.9)
    training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epocks):
        if epoch % 100 == 0:
            print(f'Epoch {epoch:04d}: MES = {mse.eval()}')
        sess.run(training_op)
    theta_value = theta.eval()

# print(theta_value)
sigma = np.std(housing.data, axis=0).reshape(-1, 1)
mu = np.mean(housing.data, axis=0).reshape(-1, 1)
th = theta_value[1:] / sigma
non_scaled_theta = np.r_[theta_value[0] - np.dot(mu.T, th), th]
print(non_scaled_theta)
print(non_scaled_theta - np.load(os.path.join(base_dir, 'theta.npy')))
