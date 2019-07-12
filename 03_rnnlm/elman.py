import os
from datetime import datetime
from functools import reduce
from math import sqrt

import numpy as np
from numpy.random import randn
from tensorflow.compat import v1 as tf

from mixins import BoardRecorderMixin


class Elman(BoardRecorderMixin):
    """Recurrent Neural Network Language Model.
    """

    @classmethod
    def create_from_text(cls, text):
        """Create RNNLM instance from text.

        Args:
            text (str): text to analyze

        Returns:
            RNNLM: Instance created
        """
        duplicate_words = text.lower().replace('.', ' .').split(' ')
        words = list(set(duplicate_words))
        corpus = [words.index(word) for word in duplicate_words]
        return cls(words, corpus)

    def __init__(self, words, corpus):
        """Create instance form corpus.

        Args:
            words (list): List of words
            corpus (list): Corpus
        """
        self.words = words
        self.vocab_size = len(words)
        self.corpus = corpus

    @property
    def data_size(self):
        if not hasattr(self, '_data_size'):
            self._data_size = len(self.corpus) - 1
        return self._data_size

    def build_graph(self, wordvec_size=100, hidden_size=100, time_size=5,
                    optimizer=None):
        """Buid tensorflow graph.

        Args:
            wordvec_size (int): Dimension of Distributed Represendations of
                the words
            hidden_size (int): Dimension of hidden layer
            time_size (int): Count to expand truncated BPTT
            optimizer: Optimizer instance. Default to tf.train.Adam
        """
        self.wordvec_size = wordvec_size
        self.hidden_size = hidden_size
        self.time_size = time_size

        self.learning_rate = tf.placeholder(tf.float32)
        incomes = tf.placeholder(
            tf.int32,
            shape=(None, time_size),
            name='incomes',
        )
        labels = tf.placeholder(
            tf.int32,
            shape=(None, time_size),
            name='labels',
        )
        prev_hs = tf.placeholder(
            tf.float32,
            shape=(None, hidden_size),
            name='prev_hs'
        )

        with tf.name_scope('time_embedding'):
            embed_W = tf.Variable(
                np.random.randn(self.vocab_size, wordvec_size) / 100,
                dtype=tf.float32,
                name='embed_W',
            )
            xs = tf.gather(embed_W, incomes)

        with tf.name_scope('time_RNN'):
            rnn_Wx = tf.Variable(
                randn(wordvec_size, hidden_size) / sqrt(wordvec_size),
                dtype=tf.float32,
                name='rnn_Wx',
            )
            rnn_Wh = tf.Variable(
                randn(hidden_size, hidden_size) / sqrt(hidden_size),
                dtype=tf.float32,
                name='rnn_Wh',
            )
            rnn_bias = tf.Variable(
                np.zeros(hidden_size),
                dtype=tf.float32,
                name='rnn_bias',
            )

            xW = tf.matmul(xs, rnn_Wx)
            hs = tf.stack(
                reduce(
                    lambda array, i: array + [
                        tf.math.tanh(
                            tf.matmul(array[-1], rnn_Wh)
                            + xW[:, 0, :]
                            + rnn_bias
                        )
                    ],
                    range(time_size - 1),
                    [
                        tf.math.tanh(
                            tf.matmul(prev_hs, rnn_Wh)
                            + xW[:, 0, :]
                            + rnn_bias
                        )
                    ]
                ),
                1,
            )

        with tf.name_scope('time_affine'):
            affine_W = tf.Variable(
                randn(hidden_size, self.vocab_size) / sqrt(hidden_size),
                dtype=tf.float32,
                name='affine_W',
            )
            affine_bias = tf.Variable(
                np.zeros(self.vocab_size),
                dtype=tf.float32,
                name='affine_bias',
            )
            logits = tf.math.add(
                tf.matmul(hs, affine_W),
                affine_bias,
                name='logits',
            )

        cee = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(labels, [-1]),
                logits=tf.reshape(logits, [-1, self.vocab_size]),
            ),
            name='CEE',
        )
        self.los_summaries = [
            tf.summary.scalar('Perplexity', tf.math.exp(cee), family='Loss'),
            tf.summary.scalar('Corss_Entorpy_Error', cee, family='Loss'),
        ]

        self.incomes = incomes
        self.labels = labels
        self.prev_hs = prev_hs
        self.next_hs = hs[:, -1, :]

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)
        self.training_op = optimizer.minimize(cee)

    def fetch_batch(self, epoch_i, batch_i, batch_size, jump, incomes, labels):
        data_size = len(self.corpus) - 1
        if not hasattr(self, '_t'):
            self._t = 0
        for b in range(batch_size):
            for t in range(self.time_size):
                incomes[b, t] = self.corpus[(b * jump + self._t) % data_size]
                labels[b, t] = self.corpus[(b * jump + self._t + 1) % data_size]
                self._t += 1
        return (incomes, labels)

    def train(self, log_dir=None, max_epoch=10000, learning_rate=0.001,
              batch_size=None, interval_sec=300, restore_step=None,
              run_metadata=False):
        """Train model.

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
            run_metadata (bool): If true, run metadata and write logs.
        """
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__),
                                   'tf_logs',
                                   datetime.utcnow().strftime('%Y%m%d%H%M%S'))
        if batch_size is None:
            batch_size = 1
        n_batches = len(self.corpus) // (batch_size * self.time_size)
        jump = (len(self.corpus) - 1) // batch_size
        if run_metadata:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            metadata = tf.RunMetadata()
        else:
            options = None
            metadata = None
        with self.open_writer(log_dir) as writer:
            with self.open_session(interval_sec=interval_sec,
                                   per_step=n_batches,
                                   restore_step=restore_step) as sess:
                incomes = np.empty([batch_size, self.time_size], dtype=int)
                labels = np.empty([batch_size, self.time_size], dtype=int)
                for b in range(batch_size):
                    incomes[b, ] = self.corpus[b*jump:b*jump+self.time_size]
                    labels[b, ] = self.corpus[b*jump+1:b*jump+self.time_size+1]
                step = restore_step or 0
                next_hs = np.zeros([batch_size, self.hidden_size])
                if restore_step is None:
                    for summary in sess.run(
                        self.los_summaries,
                        feed_dict={self.incomes: incomes[:batch_size],
                                   self.labels: labels[:batch_size],
                                   self.prev_hs: next_hs},
                    ):
                        writer.add_summary(summary, step)
                for epoch_i in range(step // self.data_size, max_epoch):
                    for batch_i in range(n_batches):
                        inc, lab = self.fetch_batch(epoch_i, batch_i,
                                                    batch_size, jump,
                                                    incomes, labels)
                        fd = {
                            self.incomes: inc,
                            self.labels: lab,
                            self.prev_hs: next_hs,
                            self.learning_rate: learning_rate,
                        }
                        _, next_hs = sess.run([self.training_op, self.next_hs],
                                              feed_dict=fd,
                                              options=options,
                                              run_metadata=metadata)
                        step += 1
                        if run_metadata:
                            writer.add_run_metadata(metadata, f'step: {step}')
                        self.record(sess, writer, step, feed_dict=fd)
                self.record(sess, writer, step, feed_dict=fd, force_write=True)
