import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1

from word2vec import Word2Vec


class CBOW(Word2Vec):
    """Wor2Vec by CBOW model.

    Attributes:
        words (list): List of words
        words_size (int): Size of `words`
        corpus (list): Corpus
        word_vectors (:obj:`WordVectors`): Results of model. You can reforence
            after call `train` method.
    """

    def get_incomes(self):
        return self.get_contexts()

    def get_labels(self):
        return self.get_targets()

    def fetch_batch(self, incomes, labels, epoch_i, batch_i, batch_size):
        if batch_size is None or self.data_size < batch_size:
            return (
                incomes,
                labels,
                self.data_size,
            )
        if not hasattr(self, 'cl'):
            self.cl = np.concatenate(
                [incomes, labels.reshape((-1, 1))],
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
        self.build_model_params(window_size, hidden_size)

        self.incomes = tfv1.placeholder(
            tf.int32,
            shape=(None, self.window_size*2),
            name='incomes',
        )
        self.labels = tfv1.placeholder(
            tf.int32,
            shape=(None,),
            name='labels',
        )
        self.batch_size = tfv1.placeholder(tf.float32, name='batch_size')

        cee = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels,
                logits=self._build_cbow_output(self.incomes),
            ), name='CEE',
        )
        self.training_op = self._build_optimize_graph(cee)
        self.los_summary = tfv1.summary.scalar('Loss', cee)

    def _build_optimize_graph(self, cee):
        optimizer = tfv1.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(cee)

    def _build_cbow_output(self, incomes):
        b, w = incomes.shape
        v, h = self.W_in.shape.as_list()
        return tf.matmul(
            tf.reshape(
                tf.reduce_mean(
                    tf.reshape(
                        tf.transpose(
                            tf.reshape(
                                tf.gather(
                                    self.W_in,
                                    tf.reshape(incomes, [-1]),
                                ), [-1, w, h],
                            ), [1, 0, 2],
                        ), [w, -1],
                    ), axis=0,
                ), [-1, h],
            ),
            self.W_out,
            name='logits',
        )
