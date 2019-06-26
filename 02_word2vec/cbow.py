import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1

from mixins import BoardRecorderMixin
from word2vec import DistributedRepresentations


class CBOW(BoardRecorderMixin):
    """Accelerated CBOW model.

    Attributes:
        words (list): List of words
        words_size (int): Size of `words`
        corpus (list): Corpus
        word_vectors (:obj:`WordVectors`): Results of model. You can reforence
            after call `train` method.
    """

    model_file_name = 'model.chpt'

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

    @property
    def data_size(self):
        if not hasattr(self, '_data_size'):
            self._data_size = len(self.corpus) - self.window_size * 2
        return self._data_size

    def fetch_batch(self, contexts, labels, epoch_i, batch_i, batch_size):
        if batch_size is None or self.data_size < batch_size:
            return (
                contexts,
                labels,
                self.data_size,
            )
        if not hasattr(self, 'cl'):
            self.cl = np.concatenate(
                [contexts, labels.reshape((-1, 1))],
                axis=1,
            )
        if batch_i == 0:
            np.random.shuffle(self.cl)
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
        if hasattr(self, '_data_size'):
            del self._data_size
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.learning_rate = tfv1.placeholder(tf.float32, name='learning_rate')

        self.W_in = tf.Variable(
            tf.random.uniform([self.words_size, self.hidden_size],
                              -1.0, 1.0, dtype=tf.float32),
            dtype=tf.float32,
            name='W_in',
        )
        self.W_out = tf.Variable(
            tf.random.uniform([self.hidden_size, self.words_size],
                              -1.0, 1.0, dtype=tf.float32),
            dtype=tf.float32,
            name='W_out',
        )
        contexts = tfv1.placeholder(
            tf.int32,
            shape=(None, self.window_size*2),
            name='contexts',
        )
        labels = tfv1.placeholder(
            tf.int32,
            shape=(None,),
            name='labels',
        )
        batch_size = tfv1.placeholder(tf.float32, name='batch_size')
        self.contexts = contexts
        self.labels = labels
        self.batch_size = batch_size

        cee = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=self._build_cbow_output(contexts),
            ), name='CEE',
        )
        self.training_op = self._build_optimize_graph(cee)
        self.los_summary = tfv1.summary.scalar('Loss', cee)

    def _build_optimize_graph(self, cee):
        optimizer = tfv1.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(cee)

    def _build_cbow_output(self, contexts):
        b, w = contexts.shape
        v, h = self.W_in.shape.as_list()
        return tf.matmul(
            tf.reshape(
                tf.reduce_mean(
                    tf.reshape(
                        tf.transpose(
                            tf.reshape(
                                tf.gather(
                                    self.W_in,
                                    tf.reshape(contexts, [-1]),
                                ), [-1, w, h],
                            ), [1, 0, 2],
                        ), [w, -1],
                    ), axis=0,
                ), [-1, h],
            ),
            self.W_out,
        )

    def train(self, log_dir=None, max_epoch=10000, learning_rate=0.001,
              batch_size=None, interval_sec=300, restore_step=None):
        """Train CBOW model.

        Args:
            log_dir (str): Log directory where log and model is saved.
            max_epoch (int): Size of epoch
            learning_rate (float): Learning rate
            batch_size (int): Batch size when using mini-batch descent method.
                If specifying a size larger then learning data or `None`,
                using batch descent.
            interfal_sec (float): Specify logging time interval in seconds.
                Default by 300.
            restore_step (int): When you specify this argument, this mixin
                resotres model for specified step.
        """
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__),
                                   'tf_logs',
                                   datetime.utcnow().strftime('%Y%m%d%H%M%S'))
        if batch_size is None:
            n_batches = 1
        else:
            n_batches = int(np.ceil(self.data_size / batch_size))
        with self.open_writer(log_dir) as writer:
            with self.open_session(interval_sec=interval_sec,
                                   per_step=n_batches,
                                   restore_step=restore_step) as sess:
                contexts = self.get_contexts()
                labels = self.get_target()
                self.word_reps = DistributedRepresentations(
                    self.words,
                    sess.run(self.W_in))
                step = restore_step or 0
                if restore_step is None:
                    writer.add_summary(
                        self.los_summary.eval(
                            feed_dict={self.contexts: contexts[:batch_size],
                                       self.labels: labels[:batch_size]},
                        ),
                        step,
                    )
                for epoch_i in range(step // self.data_size, max_epoch):
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
                        self.record(sess, writer, step, feed_dict=fd)
                        step += 1
                    self.word_reps.vecs = sess.run(self.W_in)
                self.record(sess, writer, step, feed_dict=fd, force_write=True)
