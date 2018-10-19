"""
Defining standard tensorflow layers as modules.
"""

import tensorflow as tf
from deeplearning import module
from deeplearning import tf_util as U
import numpy as np

class Input(module.Module):
    ninputs=0

class Placeholder(Input):
    def __init__(self, dtype, shape, name, default=None):
        super().__init__(name)
        assert isinstance(shape, (list, tuple))
        self.dtype = dtype
        self.shape = list(shape)
        self.phname = name
        self.default = default

    def _bsz(self):
        return self.nbatch * self.nstep

    def _build(self, inputs):
        bsz = self._bsz()
        shape = [bsz] + self.shape
        if self.default is not None:
            assert list(self.default.shape) == self.shape
            default = np.repeat(self.default[None], bsz, axis=0)
            self.ph = tf.placeholder_with_default(default, shape=shape, name=self.phname)
        else:
            self.ph = tf.placeholder(self.dtype, shape=shape, name=self.phname)
        self.placeholders.append(self.ph)
        return self.ph

class StatePlaceholder(Placeholder):
    def _bsz(self):
        return self.nbatch

class Dense(module.Module):
    ninputs = 1
    def __init__(self, name, *modules, units=1, activation=None, **kwargs):
        super().__init__(name, *modules)
        self.units = units
        self.activation = activation
        self.layer_kwargs = kwargs

    def _build(self, inputs):
        return tf.layers.dense(inputs[0],
                               self.units,
                               kernel_initializer=tf.variance_scaling_initializer(),
                               activation=self.activation,
                               name='dense',
                               **self.layer_kwargs)

class Conv2d(module.Module):
    ninputs = 1
    def __init__(self, name, *modules, filters=1, size=(3,3), strides=(1,1), padding='valid', activation=None, **kwargs):
        super().__init__(name, *modules)
        self.filters = filters
        self.size = size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.layer_kwargs = kwargs

    def _build(self, inputs):
        return tf.layers.conv2d(inputs[0],
                                filters=self.filters,
                                strides=self.strides,
                                kernel_size=self.size,
                                kernel_initializer=tf.variance_scaling_initializer(),
                                padding=self.padding,
                                activation=self.activation,
                                name='conv2d',
                                **self.layer_kwargs)

class LSTM(module.RecurrentModule):
    def __init__(self, name, *modules, nlstm=256, nlayers=1, masked=False):
        assert len(modules) == 1, "This LSTM is only designed to work with 1 input"
        modules = list(modules)
        if masked:
            modules.append(Placeholder(tf.float32, [], name=name+'_ph_mask'))

        state_modules = []
        default = np.zeros((nlstm*2,), dtype=np.float32)
        for i in range(nlayers):
            state_modules.append(StatePlaceholder(tf.float32, (nlstm*2,), name+'_ph%d'%i, default))
        super().__init__(name, *modules, state_modules=state_modules)

        self.nlayers = nlayers
        self.nlstm = nlstm
        self.masked = masked

    def _build(self, inputs, state):
        X = inputs[0]
        M = inputs[1] if self.masked else tf.zeros([self.nbatch * self.nstep])
        ms = U.batch_to_seq(M, self.nbatch, self.nstep)
        hs = U.batch_to_seq(X, self.nbatch, self.nstep)
        state_out = []
        for i in range(self.nlayers):
            hs, soi = U.lstm(hs, ms, state[i], 'lstm{}'.format(i), nh=self.nlstm)
            state_out.append(soi)
        h = U.seq_to_batch(hs)
        return h, state_out

class Flatten(module.Module):
    ninputs = 1
    def _build(self, inputs):
        return tf.layers.flatten(inputs[0])

class StopGrad(module.Module):
    def _build(self, inputs):
        return [tf.stop_gradient(i) for i in inputs]

class Softmax(module.Module):
    ninputs = 2 # logits, targets
    def _build(self, inputs):
        logits, target = inputs
        return tf.losses.softmax_cross_entropy(target, logits)
