from deeplearning.module import Module
from deeplearning.layers import Placeholder
from rl.rl_module import ActorCritic
import tensorflow as tf
from gym import spaces

class A2CLoss(Module):
    def __init__(self, name, actor_critic, vf_coef=0.5, ent_coef=0.01):
        assert isinstance(actor_critic, ActorCritic)
        self.actor = actor_critic.actor
        # Inputs are actor_critic, actions, rewards, advantages
        ac_ph = Placeholder(tf.float32, self.actor.pdtype.sample_shape(), name+'_ac_ph')
        vtarg_ph = Placeholder(tf.float32, [1], name+'_vtarg_ph')
        atarg_ph = Placeholder(tf.float32, [1], name+'_atarg_ph')
        super().__init__(name, actor_critic, ac_ph, vtarg_ph, atarg_ph)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def _build(self, inputs):
        _, vpred, ac, vtarg, atarg = inputs
        if not isinstance(self.actor.ac_space, spaces.Box):
            ac = tf.cast(ac, tf.int32)
        neglogp = self.actor.pd.neglogp(ac)
        # check that neglogp has the correct dimension. (It can be off when the action space is discrete)
        if len(neglogp.shape) < len(atarg.shape):
            neglogp = tf.expand_dims(neglogp, -1)
        assert len(neglogp.shape) == len(atarg.shape)
        assert len(vpred.shape) == len(vtarg.shape)
        self._p_loss = tf.reduce_mean(neglogp * atarg)
        self._v_loss = 0.5 * tf.reduce_mean(tf.square(vpred - vtarg))
        self._entropy = tf.reduce_mean(self.actor._entropy)
        return self._p_loss + self.vf_coef * self._v_loss - self.ent_coef * self._entropy

    def _add_run_args(self, outs, feed_dict, **flags):
        if 'p_loss' in flags and flags['p_loss']:
            outs['p_loss'] = self._p_loss
        if 'v_loss' in flags and flags['v_loss']:
            outs['v_loss'] = self._v_loss
        if 'ent_loss' in flags and flags['ent_loss']:
            outs['ent_loss'] = self._entropy

    # convenience methods
    def p_loss(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, p_loss=True)['p_Loss']

    def v_loss(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, v_loss=True)['v_loss']

    def ent_loss(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, ent_loss=True)['ent_loss']
