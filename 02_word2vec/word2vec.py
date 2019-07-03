import os
from collections import Counter
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1

from mixins import BoardRecorderMixin


class DistributedRepresentations:
    """Distributed Represendations of the words.

    Args:
        words (list): List of words
        vectors (numpy.array): Vectors encoded words

    Attributes:
        vecs (numpy.array): Vectors encoded words
        words (list): List of words
    """
    def __init__(self, words, vectors):
        self.words = words
        self.vecs = vectors

    @property
    def normalized_vecs(self):
        return self.vecs / \
            np.linalg.norm(self.vecs, axis=1).reshape(
                [self.vecs.shape[0], 1],
            )

    def inspect(self):
        rels = np.dot(self.normalized_vecs, self.normalized_vecs.T)
        printoptions = np.get_printoptions()
        np.set_printoptions(linewidth=200, precision=6)
        for word, vec in zip(self.words, rels):
            print(f'{word + ":":8s} {vec}')
        np.set_printoptions(**printoptions)

    def cos_similarity(self, x, y, eps=1e-8):
        return np.dot(x, y) / (np.linalg.norm(x) + eps) \
            / (np.linalg.norm(y) + eps)

    def words_similarity(self, word1, word2, eps=1e-8):
        x, y = [self.vecs[i]
                for i in [self.words.index(word) for word in [word1, word2]]]
        return self.cos_similarity(x, y, eps=eps)

    def most_similar(self, word, top=5):
        try:
            word_id = self.words.index(word)
        except ValueError:
            print(f"'{word}' is not found.")
            return
        print(f'\n[query]: {word}')
        word_vec = self.vecs[word_id]
        similarity = [[w, self.cos_similarity(word_vec, self.vecs[i])]
                      for i, w in enumerate(self.words) if i != word_id]
        similarity.sort(key=lambda sim: sim[1], reverse=True)
        for s in similarity[:top]:
            print(f' {s[0]}: {s[1]}')

    def analogy(self, a, b, c, top=5, answer=None):
        try:
            a_vec, b_vec, c_vec = \
                self.vecs[[self.words.index(word) for word in (a, b, c)]]
        except ValueError as err:
            print(err)
            return

        print(f'{a}:{b} = {c}:?')
        query_vec = b_vec - a_vec + c_vec
        if answer is not None:
            try:
                answer_id = self.words.index(answer)
                print(
                    f'  ==> {answer}: '
                    f'{self.cos_similarity(self.vecs[answer_id], query_vec)}'
                )
            except ValueError as err:
                print(err)
        similarity = [[w, self.cos_similarity(query_vec, self.vecs[i])]
                      for i, w in enumerate(self.words)]
        similarity.sort(key=lambda sim: sim[1], reverse=True)
        count = 0
        for s in similarity:
            if s[0] not in (a, b, c):
                print(f'  {s[0]}: {s[1]}')
                count += 1
                if top <= count:
                    print()
                    break


class Word2Vec(BoardRecorderMixin):
    """Base class for wor2vec.

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
        counter = Counter(corpus)
        self.counts = np.array([counter[i] for i in range(self.words_size)])

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

    def get_targets(self):
        return np.array(self.corpus[self.window_size:-self.window_size])

    @property
    def data_size(self):
        if not hasattr(self, '_data_size'):
            self._data_size = len(self.corpus) - self.window_size * 2
        return self._data_size

    def build_model_params(self, window_size, hidden_size):
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

    def build_model(self, window_size=1, hidden_size=5,
                    ns_count=0, ns_exponent=0.75):
        """Build CBOW graph.

        Args:
            window_size (int): Window size
            hidden_size (int): Dimension of a vector encoding the words
            ns_count (int): Number of samples using negative sampling.
                If you specify 0, this object does not use negative sampling
                and use softmax function. Default to 0.
            ns_exponent (float): Value of exponent to determine probability of
                acquiring each vocabulary using Nagative sampling. Default to
                0.75.
        """
        raise NotImplementedError

    def get_incomes(self):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def fetch_batch(self, incomes, labels, epoch_i, batch_i, batch_size):
        raise NotImplementedError

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
            n_batches = 1
        else:
            n_batches = int(np.ceil(self.data_size / batch_size))
        if run_metadata:
            options = tfv1.RunOptions(trace_level=tfv1.RunOptions.FULL_TRACE)
            metadata = tfv1.RunMetadata()
        else:
            options = None
            metadata = None
        with self.open_writer(log_dir) as writer:
            with self.open_session(interval_sec=interval_sec,
                                   per_step=n_batches,
                                   restore_step=restore_step) as sess:
                incomes = self.get_incomes()
                labels = self.get_labels()
                self.word_reps = DistributedRepresentations(
                    self.words,
                    sess.run(self.W_in))
                step = restore_step or 0
                if restore_step is None:
                    writer.add_summary(
                        self.los_summary.eval(
                            feed_dict={self.incomes: incomes[:batch_size],
                                       self.labels: labels[:batch_size]},
                        ),
                        step,
                    )
                for epoch_i in range(step // self.data_size, max_epoch):
                    for batch_i in range(n_batches):
                        c, l, b = self.fetch_batch(incomes, labels,
                                                   epoch_i, batch_i,
                                                   batch_size)
                        fd = {
                            self.incomes: c,
                            self.labels: l,
                            self.batch_size: b,
                            self.learning_rate: learning_rate,
                        }
                        sess.run(self.training_op,
                                 feed_dict=fd,
                                 options=options,
                                 run_metadata=metadata)
                        if run_metadata:
                            writer.add_run_metadata(metadata, f'step: {step}')
                        self.record(sess, writer, step, feed_dict=fd)
                        step += 1
                    self.word_reps.vecs = sess.run(self.W_in)
                self.record(sess, writer, step, feed_dict=fd, force_write=True)
