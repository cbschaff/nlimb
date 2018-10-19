"""
PPO RL algorithm.
https://arxiv.org/abs/1707.06347
"""

import numpy as np
import tensorflow as tf
from rl.losses import PPOLoss
from rl.algorithms import OnlineRLAlgorithm
from rl.runner import *
from rl import util
from deeplearning.layers import Adam, RunningNorm
from deeplearning import logger
import time

class PPO(OnlineRLAlgorithm):
    def defaults(self):
        return {
            'epochs_per_iter': 10,
            'lr': 1e-4,
            'momentum': 0.9,
            'beta2': 0.999,
            'clip_norm': 0.5,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'clip_param': 0.2,
            'gamma': 0.99,
            'lambda_': 1.0,
        }

    def __init__(self,
        logdir,
        env_fn,
        model_fn,
        nenv,
        rollout_length=1024,
        batch_size=64,
        callback=None,
        **kwargs
    ):
        defaults = self.defaults()
        for k in kwargs:
            assert k in defaults, "Unknown argument: {}".format(k)
        defaults.update(kwargs)

        super().__init__(logdir, env_fn, model_fn, nenv, rollout_length, batch_size, callback, runner_flags=['neglogp'], **defaults)

        submods = self.opt.find_submodules_by_instance(RunningNorm)
        self.rn = submods[0] if len(submods) > 0 else None
        self._time_start = time.time()
        self._t_start = self.t

    def _def_loss(self, model_fn, env):
        return PPOLoss(
                'loss',
                model_fn(env),
                vf_coef=self.args.vf_coef,
                ent_coef=self.args.ent_coef,
                clip_param=self.args.clip_param
        )

    def _def_opt(self, loss):
        return Adam(
                'opt',
                loss,
                lr=self.args.lr,
                beta1=self.args.momentum,
                beta2=self.args.beta2,
                clip_norm=self.args.clip_norm
        )

    def _before_step(self):
        logger.log("========================|  Iteration: {}  |========================".format(self.t // self.timesteps_per_step))

    def _process_rollout(self, rollout):
        data = rollout.numpy()
        util.gae(data, self.args.gamma, self.args.lambda_)
        data['atarg'] = self._norm_advantages(data['atarg'])
        return data

    def _update_model(self, data):
        if self.rn is not None:
            self._update_running_norm(data['obs'].reshape(-1, *data['obs'].shape[2:]))
        dataset, state_init = util.make_dataset(data, self.loss.is_recurrent)
        for i in range(self.args.epochs_per_iter):
            losses = []
            state = state_init
            for batch in dataset.iterate_once(self.batch_size * self.nenv):
                out = self._update(batch, state)
                state = out['state_out']
                losses.append([out['out'], out['p_loss'], out['v_loss'], out['ent_loss']])

            meanlosses = np.array(losses).mean(axis=0)
            s = 'Losses:  '
            for i,ln in enumerate(['Total', 'Policy', 'Value', 'Entropy']):
                s += ln + ': {:08f}  '.format(meanlosses[i])
            logger.log(s)
        return meanlosses

    def _after_step(self, rollout, data, losses):
        for i,s in enumerate(['Total Loss', 'Policy Loss', 'Value Loss', 'Entropy']):
            logger.logkv(s, losses[i])
        vtarg_flat = data['vtarg'].flatten()
        vpred_flat = data['vpreds'].flatten()

        logger.logkv('timesteps', self.t)
        logger.logkv('serial timesteps', self.t / self.nenv)
        logger.logkv('mean episode length', np.mean(self.runner.get_episode_lengths()))
        logger.logkv('mean episode reward', np.mean(self.runner.get_episode_rewards()))
        logger.logkv('explained var. of vtarg', util.explained_variance(vpred_flat, vtarg_flat))
        logger.logkv('fps', int((self.t - self._t_start) / (time.time() - self._time_start)))
        logger.logkv('time_elapsed', time.time() - self._time_start)
        logger.dumpkvs()


    def _update(self, batch, state):
        if self.loss.is_recurrent:
            inputs=[batch['obs'], batch['dones'], batch['actions'], batch['vtarg'], batch['atarg'], batch['neglogp'], batch['vpreds']]
        else:
            inputs=[batch['obs'], batch['actions'], batch['vtarg'], batch['atarg'], batch['neglogp'], batch['vpreds']]
        return self.opt.run(inputs=inputs, state=state, grad=False, update=True, p_loss=True, v_loss=True, ent_loss=True)

    def _norm_advantages(self, advs):
        return (advs - advs.mean()) / (advs.std() + 1e-8)

    def _update_running_norm(self, x):
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        count = x.shape[0]
        self.rn.update(mean, var, count)

    def update_lr(self, new_lr):
        self.opt.update_lr(new_lr)
        self.loss.update_clip_mult(new_lr / self.args.lr)
