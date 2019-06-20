import tensorflow as tf

from simple_cbow import SimpleCBOW


class CBOW(SimpleCBOW):
    """Accelerated CBOW model.

    Attributes:
        words (list): List of words
        words_size (int): Size of `words`
        corpus (list): Corpus
        word_vectors (:obj:`WordVectors`): Results of model. You can reforence
            after call `train` method.
    """

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
                                    tf.gather(
                                        self.W_in,
                                        tf.reshape(
                                            contexts,
                                            [-1]
                                        ),
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
            tf.gather_nd(
                tf.log(pred + self.__class__.eps),
                tf.concat(
                    [
                        tf.reshape(tf.range(tf.shape(target)[0]), [-1, 1]),
                        tf.reshape(target, (-1, 1)),
                    ],
                    axis=1,
                )
            ), name='CEE',
        ) / batch_size
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.training_op = optimizer.minimize(self.cee)
        self.cee_summary = tf.summary.scalar('CEE', self.cee)
