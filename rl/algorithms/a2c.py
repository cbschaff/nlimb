"""
A2C RL algorithm.
https://arxiv.org/abs/1602.01783
"""

import numpy as np
import tensorflow as tf
from rl.losses import A2CLoss
from rl.algorithms import OnlineRLAlgorithm
from rl.runner import *
from rl import util
from deeplearning.layers import Adam, RunningNorm
from deeplearning import logger
from collections import deque
import time


class A2C(OnlineRLAlgorithm):
    def defaults(self):
        return {
            'lr': 1e-4,
            'momentum': 0.9,
            'beta2': 0.999,
            'clip_norm': 0.5,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'gamma': 0.99,
            'lambda_': 1.0,
        }

    def __init__(self,
        logdir,
        env_fn,
        model_fn,
        nenv,
        rollout_length=32,
        batch_size=32, # ignored
        callback=None,
        **kwargs
    ):
        defaults = self.defaults()
        for k in kwargs:
            assert k in defaults, "Unknown argument: {}".format(k)
        defaults.update(kwargs)

        super().__init__(logdir, env_fn, model_fn, nenv, rollout_length, rollout_length, callback, runner_flags=[], **defaults)

        submods = self.opt.find_submodules_by_instance(RunningNorm)
        self.rn = submods[0] if len(submods) > 0 else None
        self._time_start = time.time()
        self._t_start = self.t
        self.losses = deque(maxlen=100)
        self.vpreds = deque(maxlen=1000)
        self.vtargs = deque(maxlen=1000)
        self.nsteps = 0


    def _def_loss(self, model_fn, env):
        return A2CLoss(
                'loss',
                model_fn(env),
                vf_coef=self.args.vf_coef,
                ent_coef=self.args.ent_coef
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

    def _process_rollout(self, rollout):
        data = rollout.numpy()
        util.gae(data, self.args.gamma, self.args.lambda_)
        data['atarg'] = self._norm_advantages(data['atarg'])
        return data

    def _update_model(self, data):
        if self.rn is not None:
            self._update_running_norm(data['obs'].reshape(-1, *data['obs'].shape[2:]))
        dataset, state = util.make_dataset(data, self.loss.is_recurrent)
        batch = dataset.data_map
        if self.loss.is_recurrent:
            inputs=[batch['obs'], batch['dones'], batch['actions'], batch['vtarg'], batch['atarg']]
        else:
            inputs=[batch['obs'], batch['actions'], batch['vtarg'], batch['atarg']]
        return self.opt.run(inputs=inputs, state=state, state_out=False, grad=False, update=True, p_loss=True, v_loss=True, ent_loss=True)


    def _after_step(self, rollout, data, losses):
        self.losses.append([losses['out'], losses['p_loss'], losses['v_loss'], losses['ent_loss']])
        self.vtargs.extend(list(np.array(data['vtarg']).flatten()))
        self.vpreds.extend(list(np.array(data['vpreds']).flatten()))

        self.nsteps += 1
        if self.nsteps % 100 == 0 and self.nsteps > 0:
            logger.log("========================|  Timestep: {}  |========================".format(self.t))
            meanlosses = np.mean(np.array(self.losses), axis=0)
            # Logging stats...
            for i,s in enumerate(['Total Loss', 'Policy Loss', 'Value Loss', 'Entropy']):
                logger.logkv(s, meanlosses[i])
            logger.logkv('timesteps', self.t)
            logger.logkv('serial timesteps', self.t / self.nenv)
            logger.logkv('mean episode length', np.mean(self.runner.get_episode_lengths()))
            logger.logkv('mean episode reward', np.mean(self.runner.get_episode_rewards()))
            logger.logkv('explained var. of vtarg', util.explained_variance(np.array(self.vpreds), np.array(self.vtargs)))
            logger.logkv('fps', int((self.t - self._t_start) / (time.time() - self._time_start)))
            logger.logkv('time_elapsed', time.time() - self._time_start)
            logger.dumpkvs()

    def _norm_advantages(self, advs):
        return (advs - advs.mean()) / (advs.std() + 1e-8)

    def _update_running_norm(self, x):
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        count = x.shape[0]
        self.rn.update(mean, var, count)

    def update_lr(self, new_lr):
        self.opt.update_lr(new_lr)
