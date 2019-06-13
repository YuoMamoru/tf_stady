import os

import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf


housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
mse = tf.reduce_mean(tf.square(tf.matmul(X, theta) - y), name='mse')

with tf.Session() as sess:
    theta_value, mse_value = sess.run([theta, mse])

print(f'mse: {mse_value}')
print(theta_value)
np.save(os.path.join(os.path.dirname(__file__), 'theta.npy'), theta_value)
