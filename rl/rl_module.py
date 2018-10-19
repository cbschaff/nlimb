import tensorflow as tf
from deeplearning.distributions import make_pdtype, DiagGaussianPdType
from deeplearning.module import Module
from deeplearning.layers import Placeholder
from deeplearning import tf_util as U
from gym import spaces
import numpy as np

class ValueFunction(Module):
    def _head(self, inputs):
        # Add value function function head
        return tf.layers.dense(
            inputs=inputs[0],
            units=1,
            kernel_initializer=U.normc_initializer(1.0),
            name='vf'
        )
    def _build(self, inputs):
        self.value = self._head(inputs)
        return self.value

class RLModule(Module):
    def __init__(self, name, *modules, ac_space=None, pdtype=None):
        assert ac_space is not None, "Must provide an action space."
        super().__init__(name, *modules)
        self.ac_space = ac_space
        # Option to overwrite standard pd types
        if pdtype is not None:
            self.pdtype = pdtype
        else:
            self.pdtype = make_pdtype(ac_space)

    def act(self, inputs, state=[]):
        """
        The output of RLModules should be an action.
        """
        return self.__call__(inputs, state)

class Policy(RLModule):
    def _discrete_head(self, inputs):
        return tf.layers.dense(
            inputs=inputs[0],
            units=self.pdtype.param_shape()[0],
            kernel_initializer=U.normc_initializer(0.01),
            name='pi'
        )
    def _continuous_head(self, inputs):
        """
        Policy head designed for continuous distributions.
        It makes logstd params independent of the network output and
        initialize them to 0.
        """
        param_shape = self.pdtype.param_shape()[0]
        mean = tf.layers.dense(
            inputs=inputs[0],
            units=param_shape // 2,
            kernel_initializer=U.normc_initializer(0.01),
            name='pi'
        )
        logstd = tf.get_variable(name="logstd", shape=[1, param_shape//2], initializer=tf.zeros_initializer())
        logstd = tf.tile(logstd, [self.nbatch*self.nstep, 1])
        return tf.concat([mean, logstd], axis=1)

    def _head(self, inputs):
        if isinstance(self.pdtype, DiagGaussianPdType):
            return self._continuous_head(inputs)
        else:
            return self._discrete_head(inputs)

    def _build(self, inputs):
        self.pi_params = self._head(inputs)
        self.pd = self.pdtype.pdfromflat(self.pi_params)
        self._mode = self.pd.mode()
        self._entropy = self.pd.entropy()
        ac = self.pd.sample()
        self._neglogp = self.pd.neglogp(ac)
        return ac

    def _add_run_args(self, outs, feed_dict, **flags):
        if 'mode' in flags and flags['mode']:
            outs['mode'] = self._mode
        if 'entropy' in flags and flags['entropy']:
            outs['entropy'] = self._entropy
        if 'neglogp' in flags and flags['neglogp']:
            outs['neglogp'] = self._neglogp

    # convenience methods
    def entropy(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, entropy=True)['entropy']

    def mode(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, mode=True)['mode']

    def neglogp(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, neglogp=True)['neglogp']


class ActorCritic(Module):
    ninputs=2
    def __init__(self, name, actor, critic):
        super().__init__(name, actor, critic)
        self.actor = actor
        self.critic = critic

    def _build(self, inputs):
        return inputs

    # convenience methods
    def act(self, inputs, state=[]):
        return self.run(inputs, state, state_out=False)['out'][0]

    def value(self, inputs, state=[]):
        return self.run(inputs, state, state_out=False)['out'][1]



class QFunction(RLModule):
    def __init__(self, name, *modules, ac_space=None, initial_eps=0.1):
        assert isinstance(ac_space, spaces.Discrete), "Action space must be discrete."
        super().__init__(name, *modules, ac_space=ac_space)
        self.eps = initial_eps

    def _head(self, inputs):
        # add qfunction head
        return tf.layers.dense(
            inputs=inputs[0],
            units=self.ac_space.n,
            kernel_initializer=U.normc_initializer(1.0),
            name='qvals'
        )

    def _build(self, inputs):
        self._qvals = self._head(inputs)

        # update epsilon
        self._eps = tf.get_variable('eps', initializer=tf.constant(self.eps), trainable=False)

        # epsilon greedy
        self._max_ac = tf.argmax(self._qvals, axis=-1)
        rand_ac = tf.random_uniform(self._max_ac.shape, 0, self.ac_space.n, dtype=tf.int64)
        rand_val = tf.random_uniform(self._max_ac.shape, 0, 1, dtype=tf.float32)
        return tf.where(rand_val >= self._eps, self._max_ac, rand_ac)

    def _add_run_args(self, outs, feed_dict, **flags):
        if 'qvals' in flags and flags['qvals']:
            outs['qvals'] = self._qvals
        if 'max_ac' in flags and flags['max_ac']:
            outs['max_ac'] = self._max_ac

    def update_eps(self, new_eps):
        self.eps = new_eps
        self._eps.load(new_eps)

    # convenience methods
    def qvals(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, qvals=True, max_ac=False)['qvals']

    def greedy_ac(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, qvals=False, max_ac=True)['max_ac']

class DuelingQFunction(QFunction):
    """
    Dueling DQN Architecture. https://arxiv.org/abs/1511.06581
    len(modules) can be 1 or 2.
    If 2, the first module is used to compute advantages and the second is used to compute the value function.
    If 1, the module is used to compute both.
    """
    def __init__(self, name, *modules, ac_space=None, initial_eps=0.1):
        assert len(modules) == 2 or len(modules) == 1
        if len(modules) == 2:
            super().__init__(name, *modules, ac_space=ac_space, initial_eps=initial_eps)
        else:
            super().__init__(name, modules[0], modules[0], ac_space=ac_space, initial_eps=initial_eps)

    def _head(self, inputs):
        pre_advs, pre_vf = inputs[0:2]
        advs = tf.layers.dense(
            inputs=pre_advs,
            units=self.ac_space.n,
            kernel_initializer=U.normc_initializer(1.0),
            name='advantages'
        )
        self._advs = advs - tf.expand_dims(tf.reduce_mean(advs, axis=-1), axis=-1)
        self._vf = tf.layers.dense(
            inputs=pre_vf,
            units=1,
            kernel_initializer=U.normc_initializer(1.0),
            name='value_funtion'
        )
        return self._vf + self._advs


class QFunction2(RLModule):
    """
    (state, action) -> value, instead of state -> (q_a1, q_a2, q_a3, ...)
    If discrete, 'a' should be action ids.
    """
    def __init__(self, name, state, a=None, ac_space=None, initial_eps=0.1):
        self._is_discrete = isinstance(ac_space, spaces.Discrete)
        if a is None:
            super().__init__(name, state, ac_space=ac_space)
        else:
            super().__init__(name, state, a, ac_space=ac_space)
        self.eps = initial_eps
        if self._is_discrete:
            self.n = ac_space.n

    def _head(self, sa):
        # add qfunction head
        return tf.layers.dense(
            inputs=sa,
            units=1,
            kernel_initializer=U.normc_initializer(1.0),
            reuse=tf.AUTO_REUSE,
            name='qvals'
        )

    def _build(self, inputs):
        if len(inputs) == 2:
            s, a = inputs
        else:
            s = inputs[0]
            a = None

        action_embeddings = tf.get_variable('action_embeddings', initializer=tf.constant(np.eye(self.n), dtype=tf.float32))
        if len(inputs) == 2:
            ea = tf.nn.embedding_lookup(action_embeddings, a)
        else:
            ea = tf.nn.embedding_lookup(action_embeddings, tf.zeros(shape=(self.nbatch * self.nstep), dtype=tf.int32))
        sa = tf.concat([s,ea], axis=1)
        self._qval = self._head(sa)


        if not self._is_discrete and a is not None:
            return a

        qvals = []
        for i in range(self.n):
            ai = tf.nn.embedding_lookup(action_embeddings, tf.constant(i * np.ones([self.nbatch*self.nstep]), dtype=tf.int32))
            sai = tf.concat([s,ai], axis=1)
            qvals.append(self._head(sai))
        self._qvals = tf.concat(qvals, axis=1)
        self._max_ac = tf.argmax(self._qvals, axis=1)
        self._eps = tf.get_variable('eps', initializer=tf.constant(self.eps), trainable=False)
        rand_ac = tf.random_uniform(self._max_ac.shape, 0, self.ac_space.n, dtype=tf.int64)
        rand_val = tf.random_uniform(self._max_ac.shape, 0, 1, dtype=tf.float32)

        # epsilon greedy
        return tf.where(rand_val >= self._eps, self._max_ac, rand_ac)

    def _add_run_args(self, outs, feed_dict, **flags):
        if 'qvals' in flags and flags['qvals']:
            outs['qvals'] = self._qvals
        if 'qval' in flags and flags['qval']:
            outs['qval'] = self._qval
        if 'max_ac' in flags and flags['max_ac'] and self._is_discrete:
            outs['max_ac'] = self._max_ac

    def update_eps(self, new_eps):
        self.eps = new_eps
        self._eps.load(new_eps)

    # convenience methods
    def qvals(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, qvals=True, max_ac=False)['qvals']

    def qval(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, qval=True, max_ac=False)['qval']

    def greedy_ac(self, inputs, state=[]):
        if self._is_discrete:
            return self.run(inputs, state, out=False, state_out=False, qval=False, max_ac=True)['max_ac']
