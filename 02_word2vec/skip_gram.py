import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1

from word2vec import Word2Vec


class SkipGram(Word2Vec):
    """Wor2Vec by Skip-gram model.

    Attributes:
        words (list): List of words
        vocab_size (int): Size of `words`
        corpus (list): Corpus
        word_vectors (:obj:`WordVectors`): Results of model. You can reforence
            after call `train` method.
    """

    def get_incomes(self):
        return self.get_targets()

    def get_labels(self):
        return self.get_contexts()

    def fetch_batch(self, incomes, labels, epoch_i, batch_i, batch_size):
        if batch_size is None or self.data_size < batch_size:
            return (
                incomes,
                labels,
                self.data_size,
            )
        if not hasattr(self, 'cl'):
            self.cl = np.concatenate(
                [incomes.reshape((-1, 1)), labels],
                axis=1,
            )
        if batch_i == 0:
            np.random.shuffle(self.cl)
        start_id = batch_size * batch_i
        end_id = min(start_id + batch_size, self.data_size)
        return (
            self.cl[start_id:end_id, :1].reshape((-1,)),
            self.cl[start_id:end_id, 1:],
            end_id - start_id,
        )

    def build_graph(self, window_size=1, hidden_size=5,
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
        self.build_model_params(window_size, hidden_size)

        self.incomes = tfv1.placeholder(
            tf.int32,
            shape=(None,),
            name='incomes',
        )
        self.labels = tfv1.placeholder(
            tf.int32,
            shape=(None, self.window_size*2),
            name='labels',
        )
        logits, hidden = self._build_skip_gram(self.incomes)

        cee = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.labels, [-1]),
                logits=tf.reshape(logits, [-1, self.vocab_size]),
            ), name='CEE',
        )
        if ns_count < 1:
            self.training_op = self._build_optimize_graph(cee)
        else:
            self.training_op = self._build_ns_optimize_graph(hidden,
                                                             self.labels,
                                                             ns_count,
                                                             ns_exponent)
        self.los_summary = tfv1.summary.scalar('Loss', cee)

    def _build_skip_gram(self, incomes):
        hidden = tf.gather(self.W_in, incomes, name='hidden')
        return (
            tf.multiply(
                tf.matmul(hidden, self.W_out),
                tf.ones([self.window_size * 2, 1, 1], dtype=tf.float32),
                name='logits',
            ),
            hidden,
        )

    def _build_optimize_graph(self, cee):
        optimizer = tfv1.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(cee)

    def _build_ns_optimize_graph(self, hidden, labels, ns_count, ns_exponent):
        prob = np.power(np.log(self.counts), ns_exponent, dtype=np.float32)
        prob_tensor = tf.multiply(
            prob,
            tf.reduce_prod(
                tf.one_hot(
                    labels,
                    self.vocab_size,
                    on_value=0.0,
                    off_value=1.0,
                    dtype=tf.float32,
                ),
                axis=1,
            ),
            name='probability',
        )
        positive_labels = tf.reshape(labels, (-1, self.window_size*2))
        negative_samples = tf.random.categorical(prob_tensor,
                                                 ns_count,
                                                 dtype=tf.int32,
                                                 name='nagative_sample')
        sampled_W_out = tf.transpose(
            tf.gather(
                self.W_out,
                tf.concat(
                    [positive_labels, negative_samples],
                    axis=1,
                    name='sample',
                ),
                axis=1,
            ),
            [1, 0, 2],
            name='sampled_W_out',
        )
        sigmoid_cross_entorpy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.concat(
                    [tf.ones_like(positive_labels, dtype=tf.float32),
                     tf.zeros_like(negative_samples, dtype=tf.float32)],
                    axis=1,
                    name='sampled_labels',
                ),
                logits=tf.reshape(
                    tf.matmul(
                        tf.reshape(hidden, [-1, 1, self.hidden_size]),
                        sampled_W_out,
                    ),
                    [-1, self.window_size * 2 + ns_count],
                    name='sampled_logits',
                )
            ),
            name='negative_sampling_loss',
        )
        tfv1.summary.scalar('Negative_Sampling_Loss',
                            sigmoid_cross_entorpy,
                            family='Loss')
        optimizer = tfv1.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(sigmoid_cross_entorpy)
