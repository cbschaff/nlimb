"""
Defining standard tensorflow optimizers as modules.
"""

import tensorflow as tf
from deeplearning import module
from deeplearning import tf_util as U

class SGD(module.Optimizer):
    ninputs = 1
    def __init__(self, name, loss, lr=1e-4, momentum=0.0, clip_norm=None):
        super().__init__(name, loss)
        self.lr = lr
        self.momentum = momentum
        self.clip_norm = clip_norm

    def _build(self, loss):
        # ops for updating the learning rate
        self._lr = tf.Variable(self.lr, name='lr', trainable=False)
        self._lr_placeholder = tf.placeholder(tf.float32, shape=(), name='lr_ph')
        self._update_lr = self._lr.assign(self._lr_placeholder)

        params = self.trainable_variables()
        self._flatgrad = U.flatgrad(loss, params, self.clip_norm)
        grads = tf.gradients(loss, params)
        if self.clip_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_norm)
        opt = tf.train.MomentumOptimizer(self.lr, momentum=self.momentum)
        return grads, opt.apply_gradients(list(zip(grads, params)))

    def _add_run_args(self, outs, feed_dict, **flags):
        super()._add_run_args(outs, feed_dict, **flags)
        if 'flatgrad' in flags and flags['flatgrad']:
            outs['flatgrad'] = self._flatgrad

    def update_lr(self, new_lr):
        self.lr = new_lr
        sess = tf.get_default_session()
        sess.run(self._update_lr, feed_dict={self._lr_placeholder:self.lr})

    # convenience method
    def flatgrad(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, flatgrad=True)['flatgrad']


class Adam(module.Optimizer):
    ninputs = 1
    def __init__(self, name, loss, lr=1e-4, beta1=0.9, beta2=0.999, clip_norm=None):
        super().__init__(name, loss)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.clip_norm = clip_norm

    def _build(self, loss):
        # ops for updating the learning rate
        self._lr = tf.Variable(self.lr, name='lr', trainable=False)
        self._lr_placeholder = tf.placeholder(tf.float32, shape=(), name='lr_ph')
        self._update_lr = self._lr.assign(self._lr_placeholder)

        params = self.trainable_variables()
        self._flatgrad = U.flatgrad(loss, params, self.clip_norm)
        grads = tf.gradients(loss, params)
        if self.clip_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_norm)
        opt = tf.train.AdamOptimizer(self.lr, beta1=self.beta1, beta2=self.beta2)
        return grads, opt.apply_gradients(list(zip(grads, params)))

    def _add_run_args(self, outs, feed_dict, **flags):
        super()._add_run_args(outs, feed_dict, **flags)
        if 'flatgrad' in flags and flags['flatgrad']:
            outs['flatgrad'] = self._flatgrad

    def update_lr(self, new_lr):
        self.lr = new_lr
        sess = tf.get_default_session()
        sess.run(self._update_lr, feed_dict={self._lr_placeholder:self.lr})

    # convenience method
    def flatgrad(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, flatgrad=True)['flatgrad']
