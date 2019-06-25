import os
import time

import tensorflow as tf
from tensorflow.compat import v1 as tfv1


class BoardRecorderMixin:
    """Mixin to store log on tensorboard and to store model.

    When this mixin, you should call `open_writer()` and `open_session()`
    in this order.

    Attributes:
        saver (tensorflow.comapt.v1.train.Saver): Saver object to store model.
        summary (tf.compat.v1.Tensor): scalar `Tensor` of type `string`
            containing the serialized `Summary` protocol.
    """

    model_file_name = 'model.chpt'

    def build_step_time_reocrder(self):
        self._last_time = tfv1.placeholder(tf.float64, name='last_time')
        self._currnet_time = tfv1.placeholder(tf.float64, name='current_time')
        self._step_run = tfv1.placeholder(tf.float64, name='step_run')
        self._per_step = tfv1.placeholder(tf.float64, name='per_step')
        tfv1.summary.scalar(
            'Step_Time',
            (self._currnet_time - self._last_time)
            / self._step_run * self._per_step,
        )

    def open_writer(self, log_dir):
        """Create `FileWriter` and `Saver` ojbect.

        Created `Saver` object map `saver` attribute of this instance.

        Args:
            log_dir (str): Log directory where log and model is saved.

        Returns:
            tensorflow.compat.v1.summary.FileWriter: `FileWriter` object.
        """
        self.model_path = os.path.join(log_dir, self.model_file_name)
        self.saver = tfv1.train.Saver()
        return tfv1.summary.FileWriter(log_dir, tfv1.get_default_graph())

    def open_session(self, interval_sec=300.0, per_step=1, restore_step=None):
        """Create `Session` object and start tensorflow session.

        Args:
            interfal_sec (float): Optional. Specify logging time interval in
                seconds. Default to 300.
            per_step (int): Optional. When you specify this argument, this
                mixin records time taken to execute specified number of step.
            restore_step (int): Optional. When you specify this argument,
                this mixin resotres model for specified step.
        """
        self.interval = interval_sec
        self.per_step = per_step
        self.last_step = restore_step or 0
        self.build_step_time_reocrder()
        self.summary = tfv1.summary.merge_all()
        init = tfv1.global_variables_initializer()
        sess = tfv1.Session()
        if restore_step is None:
            sess.run(init)
        else:
            self.saver.restore(sess, f'{self.model_path}-{restore_step}')
        self.next_recording = time.time() + self.interval
        self.last_recording = time.time()
        return sess

    def record(self, sess, writer, step, feed_dict={}, force_write=False):
        """Loggin summary on tensorboard and save model.

        Args:
            sess (tensorflow.compat.v1.Session): Session that executed.
            writer (tensorflow.compat.v1.summary.FileWriter): FileWrite to
                use to write log on tensorboard.
            step (int): Global step count.
            feed_dict (dit): Feed dictionary to use to evaluate tensor.
            force_write (bool): If specify `True`, force saving of logs and
                model. Default to `False`.
        """
        current_time = time.time()
        if (not force_write) and current_time < self.next_recording:
            return
        summary = self.summary.eval(
            feed_dict={
                self._last_time: self.last_recording,
                self._currnet_time: current_time,
                self._step_run: step - self.last_step,
                self._per_step: self.per_step,
                **feed_dict,
            },
        )
        writer.add_summary(summary, step)
        self.saver.save(sess, self.model_path, global_step=step)
        self.next_recording += self.interval
        self.last_recording = time.time()
        self.last_step = step
