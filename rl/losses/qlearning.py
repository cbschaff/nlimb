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

class QLearningLoss(Module):
    def __init__(self, name, qvals, next_qvals, qtarg, gamma=0.9, use_huber_loss=True):
        assert isinstance(qvals, (QFunction, QFunction2))
        assert isinstance(qtarg, (QFunction, QFunction2))
        self.qvals = qvals
        self.next_qvals = next_qvals
        self.qtarg = qtarg
        self.gamma = gamma
        self.use_huber_loss = use_huber_loss
        self.r = Placeholder(tf.float32, [1], name+'_r')
        self.ac = Placeholder(tf.float32, [], name=name+"_ac")
        self.done = Placeholder(tf.float32, [], name=name+"_done")
        self.weights = Placeholder(tf.float32, [1], name=name+"_iw", default=np.ones([1], dtype=np.float32))
        super().__init__(name, qvals, next_qvals, qtarg, self.r, self.ac, self.done, self.weights)

    def _build(self, inputs):
        _, _, _, r, ac, done, weights = inputs
        ac = tf.one_hot(tf.cast(ac, tf.int32), self.qvals.ac_space.n)
        ac_targ = tf.one_hot(self.next_qvals._max_ac, self.next_qvals.ac_space.n)
        assert ac_targ.shape == self.qtarg._qvals.shape
        assert ac.shape == self.qvals._qvals.shape
        qtarg = tf.reduce_sum(ac_targ * self.qtarg._qvals, axis=-1, keepdims=True)
        # don't let gradients go to the target netowrk
        qtarg = tf.stop_gradient(qtarg)
        qval = tf.reduce_sum(ac * self.qvals._qvals, axis=-1, keepdims=True)
        assert qtarg.shape == qval.shape
        assert len(r.shape) == len(qval.shape)
        if len(done.shape) < len(qtarg.shape):
            done = tf.expand_dims(done, -1)
        assert len(done.shape) == len(qtarg.shape)
        self._td_err = r + (1.0 - done) * self.gamma * qtarg - qval
        assert weights.shape == self._td_err.shape
        if self.use_huber_loss:
            self._loss = tf.reduce_mean(weights * huber_loss(self._td_err))
        else:
            self._loss = 0.5 * tf.reduce_mean(weights * tf.square(self._td_err))
        return self._loss

    def _add_run_args(self, outs, feed_dict, **flags):
        if 'td' in flags and flags['td']:
            outs['td'] = self._td_err

    # convenience methods
    def td(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, td=True)['td']
