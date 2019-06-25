import os
import time

import tensorflow as tf
from tensorflow.compat import v1 as tfv1


class BoardRecorderMixin:

    model_file_name = 'model.chpt'

    def build(self):
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
        self.model_path = os.path.join(log_dir, self.model_file_name)
        self.saver = tfv1.train.Saver()
        return tfv1.summary.FileWriter(log_dir, tfv1.get_default_graph())

    def open_session(self, interval_sec=300.0, per_step=1, restore_step=None):
        self.interval = interval_sec
        self.per_step = per_step
        self.last_step = restore_step or 0
        self.build()
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

    def record(self, sess, writer, step, feed_dict={}):
        current_time = time.time()
        if current_time < self.next_recording:
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
