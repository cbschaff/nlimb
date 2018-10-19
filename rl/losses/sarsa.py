"""
Double DQN loss.
https://arxiv.org/abs/1509.06461
"""

from deeplearning.module import Module
from deeplearning.layers import Placeholder
from deeplearning.tf_util import huber_loss
from rl.rl_module import QFunction, QFunction2
import tensorflow as tf
import numpy as np
from gym import spaces

class SARSALoss(Module):
    def __init__(self, name, qvals, next_qvals, gamma=0.9, use_huber_loss=True):
        assert isinstance(qvals, (QFunction, QFunction2))
        assert isinstance(next_qvals, (QFunction, QFunction2))
        self.qvals = qvals
        self.next_qvals = next_qvals
        self.gamma = gamma
        self.use_huber_loss = use_huber_loss
        self.r = Placeholder(tf.float32, [1], name+'_r')
        self.done = Placeholder(tf.float32, [], name=name+"_done")
        self.ac = Placeholder(tf.float32, [], name=name+"_ac")
        self.next_ac = Placeholder(tf.float32, [], name=name+"_nac")
        super().__init__(name, qvals, self.ac, next_qvals, self.next_ac, self.r, self.done)

    def _build(self, inputs):
        _, ac, _, nac, r, done = inputs
        ac = tf.one_hot(tf.cast(ac, tf.int32), self.qvals.ac_space.n)
        nac = tf.one_hot(tf.cast(nac, tf.int32), self.qvals.ac_space.n)
        assert ac.shape == self.qvals._qvals.shape
        q = tf.reduce_sum(ac * self.qvals._qvals, axis=-1, keepdims=True)
        assert nac.shape == self.next_qvals._qvals.shape
        qnext = tf.reduce_sum(nac * self.next_qvals._qvals, axis=-1, keepdims=True)

        # don't let gradients go to the target
        qnext = tf.stop_gradient(qnext)
        assert q.shape == qnext.shape
        assert len(r.shape) == len(q.shape)
        if len(done.shape) < len(q.shape):
            done = tf.expand_dims(done, -1)
        assert len(done.shape) == len(q.shape)
        self._td_err = r + (1.0 - done) * self.gamma * qnext - q
        if self.use_huber_loss:
            self._loss = tf.reduce_mean(huber_loss(self._td_err))
        else:
            self._loss = 0.5 * tf.reduce_mean(tf.square(self._td_err))
        return self._loss

    def _add_run_args(self, outs, feed_dict, **flags):
        if 'td' in flags and flags['td']:
            outs['td'] = self._td_err

    # convenience methods
    def td(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, td=True)['td']
