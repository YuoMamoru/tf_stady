import os
from datetime import datetime

import numpy as np
import tensorflow as tf


class SimpleCBOW:
    """Simple CBOW model.

    Attributes:
        words (list): List of words
        words_size (int): Size of `words`
        corpus (list): Corpus
        word_vectors (:obj:`WordVectors`): Results of model. You can reforence
            after call `train` method.
    """

    class WordVectors:
        """Result Structure of Word2Vec (CBOW).

        Args:
            words (list): List of words
            vectors (numpy.array): Vectors encoded words
            cee (float): Cross entropy error

        Attributes:
            cee (float): Cross entropy error
            vecs (numpy.array): Vectors encoded words
            words (list): List of words
        """
        def __init__(self, words, vectors, cee):
            self.words = words
            self.cee = cee
            self.vecs = vectors

        @property
        def normalized_vecs(self):
            return self.vecs / \
                np.linalg.norm(self.vecs, axis=1).reshape(
                    [self.vecs.shape[0], 1],
                )

        def inspect(self):
            rels = np.dot(self.normalized_vecs, self.normalized_vecs.T)
            np.set_printoptions(linewidth=200)
            print(self.cee)
            for word, vec in zip(self.words, rels):
                print(f'{word + ":":8s} {vec}')

    @classmethod
    def create_from_text(cls, text):
        """Create SimpleCBOW instance form text.

        Args:
            text (str): text to analyze

        Returns:
            SimpleCBOW: Instance crated
        """
        duplicate_words = text.lower().replace('.', ' .').split(' ')
        words = list(set(duplicate_words))
        corpus = [words.index(word) for word in duplicate_words]
        return cls(words, corpus)

    def __init__(self, words, corpus):
        """Create SimpleCBOW instance form corpus.

        Args:
            words (list): List of words
            corpus (list): Corpus
        """
        self.words = words
        self.words_size = len(words)
        self.corpus = corpus

    def get_contexts(self):
        return np.r_[
            np.array([
                self.corpus[i:(i-self.window_size*2 or len(self.corpus))]
                for i in range(self.window_size)
            ]),
            np.array([
                self.corpus[i:(i-self.window_size*2 or len(self.corpus))]
                for i in range(self.window_size+1, self.window_size*2+1)
            ]),
        ].T

    def get_target(self):
        return np.array(self.corpus[self.window_size:-self.window_size])

    def fetch_batch(self, contexts, labels, epoch_i, batch_i, batch_size):
        if batch_size is None or contexts.shape[0] < batch_size:
            return (
                contexts,
                labels,
                contexts.shape[0],
            )
        if batch_i == 0:
            if epoch_i == 0:
                self.cl = np.concatenate(
                    [contexts, labels.reshape((-1, 1))],
                    axis=1,
                )
                self.data_size = self.cl.shape[0]
            tf.random.shuffle(self.cl)
        start_id = batch_size * batch_i
        end_id = min(start_id + batch_size, self.data_size)
        return (
            self.cl[start_id:end_id, :-1],
            self.cl[start_id:end_id, -1:].reshape((-1,)),
            end_id - start_id,
        )

    def build_model(self, window_size=1, hidden_size=5):
        """Build CBOW graph.

        Args:
            window_size (int): Window size
            hidden_size (int): Dimension of a vector encoding the words
        """
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self.W_in = tf.Variable(
            tf.random_uniform([self.words_size, self.hidden_size],
                              -1.0, 1.0, dtype=tf.float32),
            dtype=tf.float32,
            name='W_in',
        )
        self.W_out = tf.Variable(
            tf.random_uniform([self.hidden_size, self.words_size],
                              -1.0, 1.0, dtype=tf.float32),
            dtype=tf.float32,
            name='W_out',
        )
        self.A_in = tf.constant(1 / (window_size * 2),
                                shape=[1, window_size * 2],
                                dtype=tf.float32)
        contexts = tf.placeholder(
            tf.int32,
            shape=(None, self.window_size*2),
            name='contexts',
        )
        target = tf.placeholder(
            tf.int32,
            shape=(None,),
            name='labels',
        )
        batch_size = tf.placeholder(tf.float32, name='batch_size')
        self.contexts = contexts
        self.labels = target
        self.batch_size = batch_size

        b, w = contexts.shape
        v, h = self.W_in.shape.as_list()
        pred = tf.nn.softmax(
            tf.matmul(
                tf.reshape(
                    tf.matmul(
                        self.A_in,
                        tf.reshape(
                            tf.transpose(
                                tf.reshape(
                                    tf.matmul(
                                        tf.reshape(
                                            tf.one_hot(
                                                contexts,
                                                self.words_size,
                                                dtype=tf.float32,
                                            ),
                                            [-1, v]
                                        ),
                                        self.W_in,
                                    ), [-1, w, h],
                                ), [1, 0, 2],
                            ), [w, -1],
                        ),
                    ),
                    [-1, h],
                ),
                self.W_out,
            ),
            name='predictions'
        )
        self.cee = - tf.reduce_sum(
            tf.one_hot(
                target,
                self.words_size,
                dtype=tf.float32,
            ) * tf.log(pred), name='CEE'
        ) / batch_size
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_op = optimizer.minimize(self.cee)
        self.cee_summary = tf.summary.scalar('CEE', self.cee)

    def train(self, log_dir=None, max_epoch=10000, learning_rate=0.001,
              batch_size=None):
        """Train CBOW model.

        Args:
            log_dir (str): Log directory of Tensorflow
            max_epoch (int): Size of epoch
            learning_rate (float): Learning rate
            batch_size (int): Batch size when using mini-batch descent method.
                If specifying a size larger then learning data or `None`, using
                batch descent.
        """
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__),
                                   'tf_logs',
                                   datetime.utcnow().strftime('%Y%m%d%H%M%S'))
        with tf.summary.FileWriter(log_dir, tf.get_default_graph()) as writer:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init)
                contexts = self.get_contexts()
                labels = self.get_target()
                feed_dict = {
                    self.contexts: contexts,
                    self.labels: labels,
                    self.batch_size: contexts.shape[0],
                    self.learning_rate: learning_rate,
                }
                self.word_vectors = \
                    self.WordVectors(self.words,
                                     *sess.run([self.W_in, self.cee],
                                               feed_dict=feed_dict,))
                n_batches = 1 if batch_size is None \
                    else int(np.ceil(len(labels) / batch_size))
                for epoch_i in range(max_epoch):
                    summary_str = self.cee_summary.eval(feed_dict=feed_dict)
                    writer.add_summary(summary_str, epoch_i)
                    if epoch_i % 500 == 0:
                        vectors, cee = sess.run([self.W_in, self.cee],
                                                feed_dict=feed_dict)
                        saver.save(sess,
                                   os.path.join(os.path.dirname(__file__),
                                                'models/model.chpt'))
                        print(f'Epoch {epoch_i:04d}: CEE = {cee}')
                        self.word_vectors.vecs = vectors
                        self.word_vectors.cee = cee
                    for batch_i in range(n_batches):
                        c, l, b = self.fetch_batch(contexts, labels,
                                                   epoch_i, batch_i,
                                                   batch_size)
                        fd = {
                            self.contexts: c,
                            self.labels: l,
                            self.batch_size: b,
                            self.learning_rate: learning_rate,
                        }
                        sess.run(self.training_op, feed_dict=fd)
                self.word_vectors.vecs, self.word_vectors.cee = \
                    sess.run([self.W_in, self.cee], feed_dict=feed_dict)
