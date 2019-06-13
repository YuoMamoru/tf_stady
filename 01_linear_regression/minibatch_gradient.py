from datetime import datetime
import os

import numpy as np
from scipy import stats
from sklearn.datasets import fetch_california_housing
import tensorflow as tf


housing = fetch_california_housing()
m, n = housing.data.shape
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)),
                                      stats.zscore(housing.data)]

n_epocks = 100
learning_rate = 0.001
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
np.random.seed(0)
tf.random.set_random_seed(0)
row_ids = np.random.permutation(m)
calc_grad = False
base_dir = os.path.dirname(__file__)
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = os.path.join(base_dir, 'tf_logs')
logdir = os.path.join(root_logdir, now)


def fetch_batch(batch_index, batch_size):
    if batch_index == 0:
        np.random.shuffle(row_ids)
    start_id = batch_size * batch_index
    end_id = start_id + batch_size
    return (scaled_housing_data_plus_bias[start_id:end_id],
            housing.target[start_id:end_id].reshape(-1, 1))


X = tf.placeholder(tf.float32, shape=(None, n + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
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
mse_summary = tf.summary.scalar('MSE', mse)
with tf.summary.FileWriter(logdir, tf.get_default_graph()) as file_writer:

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epocks):
            summary_str = mse_summary.eval(
                feed_dict={X: scaled_housing_data_plus_bias,
                        y: housing.target.reshape(-1, 1)})
            file_writer.add_summary(summary_str, epoch * n_batches)
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(base_dir,
                                                        'models/tmp_model.chpt'))
                # print('Epoch {0:04d}: MES = {1}'.format(
                #     epoch,
                #     mse.eval(feed_dict={X: scaled_housing_data_plus_bias,
                #                         y: housing.target.reshape(-1, 1)})
                # ))
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        theta_value = theta.eval()
        save_path = saver.save(sess, os.path.join(base_dir,
                                                'models/final_model.chpt'))

# print(theta_value)
sigma = np.std(housing.data, axis=0).reshape(-1, 1)
mu = np.mean(housing.data, axis=0).reshape(-1, 1)
th = theta_value[1:] / sigma
non_scaled_theta = np.r_[theta_value[0] - np.dot(mu.T, th), th]
print(non_scaled_theta)
print(non_scaled_theta - np.load(os.path.join(base_dir, 'theta.npy')))
