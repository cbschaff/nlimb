"""
PPO Surrogate loss function.
https://arxiv.org/abs/1707.06347
"""

from deeplearning.module import Module
from deeplearning.layers import Placeholder
from rl.rl_module import ActorCritic
import tensorflow as tf
from gym import spaces

class PPOLoss(Module):
    def __init__(self, name, ac, vf_coef=0.5, ent_coef=0.01, clip_param=0.2):
        assert isinstance(ac, ActorCritic)
        self.actor_critic = ac
        ac_ph = Placeholder(tf.float32, ac.actor.pdtype.sample_shape(), name+'_ac_ph')
        vtarg_ph = Placeholder(tf.float32, [1], name+'_vtarg_ph')
        atarg_ph = Placeholder(tf.float32, [1], name+'_atarg_ph')
        oldnlp_ph = Placeholder(tf.float32, [], name+'_oldnlp_ph')
        oldvp_ph = Placeholder(tf.float32, [1], name+'_oldvp_ph')
        super().__init__(name, ac, ac_ph, vtarg_ph, atarg_ph, oldnlp_ph, oldvp_ph)
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_param = clip_param
        self.clip_mult = 1.0

    def _build(self, inputs):
        _, vpred, ac, vtarg, atarg, old_neglogp, old_vpred = inputs
        if not isinstance(self.actor_critic.actor.ac_space, spaces.Box):
            ac = tf.cast(ac, tf.int32)
        # We need to scale the clip param with learning rate. The clip param controls
        # how much we allow our policy to change each iteration,
        # so if we want to slow down the rate at which our policy changes,
        # we need to change this parameter in addition to the learning rate.
        self._clip_mult = tf.Variable(1.0, trainable=False, name='clip_mult')

        clip_param = self.clip_param * self._clip_mult

        logp = -1 * self.actor_critic.actor.pd.neglogp(ac)
        if len(logp.shape) < len(old_neglogp.shape):
            logp = tf.expand_dims(logp, -1)
        assert len(logp.shape) == len(old_neglogp.shape)
        ratio = tf.exp(logp + old_neglogp) # pnew / pold
        # check that ratio has the correct dimension. (It can be off when the action space is discrete)
        if len(ratio.shape) < len(atarg.shape):
            ratio = tf.expand_dims(ratio, -1)
        assert len(ratio.shape) == len(atarg.shape)

        # PPO's pessimistic surrogate (L^CLIP)
        ploss1 = ratio * atarg
        ploss2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
        self._p_loss = -1. * tf.reduce_mean(tf.minimum(ploss1, ploss2))

        # we do the same clipping-based trust region for the value function
        assert len(vpred.shape) == len(vtarg.shape)
        assert len(vpred.shape) == len(old_vpred.shape)
        vfloss1 = tf.square(vpred - vtarg)
        vpredclipped = old_vpred + tf.clip_by_value(vpred - old_vpred, -clip_param, clip_param)
        vfloss2 = tf.square(vpredclipped - vtarg)
        self._v_loss = .5 * tf.reduce_mean(tf.maximum(vfloss1, vfloss2))

        # Entropy penalty
        self._entropy = tf.reduce_mean(self.actor_critic.actor._entropy)

        return self._p_loss + self.vf_coef * self._v_loss - self.ent_coef * self._entropy

    def update_clip_mult(self, new_mult):
        self.clip_mult = new_mult
        self._clip_mult.load(new_mult)

    def _add_run_args(self, outs, feed_dict, **flags):
        if 'p_loss' in flags and flags['p_loss']:
            outs['p_loss'] = self._p_loss
        if 'v_loss' in flags and flags['v_loss']:
            outs['v_loss'] = self._v_loss
        if 'ent_loss' in flags and flags['ent_loss']:
            outs['ent_loss'] = self._entropy

    # convenience methods
    def p_loss(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, p_loss=True)['p_loss']

    def v_loss(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, v_loss=True)['v_loss']

    def ent_loss(self, inputs, state=[]):
        return self.run(inputs, state, out=False, state_out=False, ent_loss=True)['ent_loss']
