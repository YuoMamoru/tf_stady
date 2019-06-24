from datetime import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1

from word2vec import DistributedRepresentations


class CBOW:
    """Accelerated CBOW model.

    Attributes:
        words (list): List of words
        words_size (int): Size of `words`
        corpus (list): Corpus
        word_vectors (:obj:`WordVectors`): Results of model. You can reforence
            after call `train` method.
    """

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
        if not hasattr(self, 'cl'):
            self.cl = np.concatenate(
                [contexts, labels.reshape((-1, 1))],
                axis=1,
            )
            self.data_size = self.cl.shape[0]
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
        target = tfv1.placeholder(
            tf.int32,
            shape=(None,),
            name='labels',
        )
        batch_size = tfv1.placeholder(tf.float32, name='batch_size')
        self.contexts = contexts
        self.labels = target
        self.batch_size = batch_size

        b, w = contexts.shape
        v, h = self.W_in.shape.as_list()
        logits = tf.matmul(
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
        self.cee = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target,
                logits=logits,
            ), name='CEE',
        )
        optimizer = tfv1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_op = optimizer.minimize(self.cee)
        self.cee_summary = tfv1.summary.scalar('CEE', self.cee)

    def train(self, log_dir=None, max_epoch=10000, learning_rate=0.001,
              batch_size=None, restore_epoch=None):
        """Train CBOW model.

        Args:
            log_dir (str): Log directory of Tensorflow
            max_epoch (int): Size of epoch
            learning_rate (float): Learning rate
            batch_size (int): Batch size when using mini-batch descent method.
                If specifying a size larger then learning data or `None`,
                using batch descent.
        """
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__),
                                   'tf_logs',
                                   datetime.utcnow().strftime('%Y%m%d%H%M%S'))
        with tfv1.summary.FileWriter(log_dir,
                                     tfv1.get_default_graph()) as writer:
            init = tfv1.global_variables_initializer()
            saver = tfv1.train.Saver()
            with tfv1.Session() as sess:
                if restore_epoch is None:
                    sess.run(init)
                    first_epoch = 0
                else:
                    saver.restore(
                        sess,
                        os.path.join(log_dir, f'model.chpt-{restore_epoch}'),
                    )
                    first_epoch = restore_epoch
                contexts = self.get_contexts()
                labels = self.get_target()
                self.word_reps = DistributedRepresentations(
                    self.words,
                    sess.run(self.W_in))
                n_batches = 1 if batch_size is None \
                    else int(np.ceil(len(labels) / batch_size))
                for epoch_i in range(first_epoch, max_epoch):
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
                        if batch_i == 0:
                            summary_str = self.cee_summary.eval(feed_dict=fd)
                            writer.add_summary(summary_str, epoch_i)
                            if epoch_i % 1 == 0:
                                vectors, cee = sess.run([self.W_in, self.cee],
                                                        feed_dict=fd)
                                saver.save(
                                    sess,
                                    os.path.join(log_dir, 'model.chpt'),
                                    global_step=epoch_i,
                                )
                                print(f'Epoch {epoch_i:04d}: CEE = {cee}')
                                self.word_reps.vecs = vectors
                        sess.run(self.training_op, feed_dict=fd)
                self.word_reps.vecs = sess.run(self.W_in)
