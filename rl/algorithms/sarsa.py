"""
A2C RL algorithm.
https://arxiv.org/abs/1602.01783
"""

import numpy as np
import tensorflow as tf
from rl.losses import SARSALoss
from rl.algorithms import OnlineRLAlgorithm
from rl.runner import *
from rl import util
from deeplearning.layers import Adam, RunningNorm
from deeplearning.schedules import LinearSchedule
from deeplearning import logger
import time


class SARSA(OnlineRLAlgorithm):
    def defaults(self):
        return {
            'lr': 1e-4,
            'momentum': 0.9,
            'beta2': 0.999,
            'clip_norm': 10.,
            'huber_loss': True,
            'gamma': 0.99,
            'epochs': 1,
            'exploration_timesteps': int(1e6),
            'final_eps': 0.02,
            'target_update_freq': int(1e4),
            'use_target_network': True,
        }

    def __init__(self,
        logdir,
        env_fn,
        model_fn,
        nenv,
        rollout_length=32,
        batch_size=32,
        callback=None,
        **kwargs
    ):
        defaults = self.defaults()
        for k in kwargs:
            assert k in defaults, "Unknown argument: {}".format(k)
        defaults.update(kwargs)

        super().__init__(logdir, env_fn, model_fn, nenv, rollout_length, batch_size, callback, runner_flags=[], **defaults)

        if self.args.use_target_network:
            self.target_sync = tf.group([tf.assign(v1,v2) for v1,v2 in zip(self.loss.next_qvals.variables(), self.loss.qvals.variables())])
            self.last_target_sync = (self.t // self.args.target_update_freq) * self.args.target_update_freq
        self.eps_schedule = LinearSchedule(int(self.args.exploration_timesteps), self.args.final_eps, 1.0)

        submods = self.opt.find_submodules_by_instance(RunningNorm)
        self.rn = submods[0] if len(submods) > 0 else None
        self._time_start = time.time()
        self._t_start = self.t


    def _def_loss(self, model_fn, env):
        qnext = model_fn(env)
        if self.args.use_target_network:
            qnext.build('target', self.nenv, self.batch_size, trainable=False)
        loss = SARSALoss(
                'loss',
                model_fn(env),
                qnext,
                gamma=self.args.gamma,
                use_huber_loss=self.args.huber_loss
        )
        assert not loss.is_recurrent
        self._loss_names = ['TD']
        self._loss_keys = ['out']
        return loss


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
        if self.args.use_target_network:
            if self.t == 0 or self.t - self.last_target_sync > self.args.target_update_freq:
                self.target_sync.run()
                self.last_target_sync = self.t
        self.actor.update_eps(self.eps_schedule.value(self.t))

    def _process_rollout(self, rollout):
        data = rollout.numpy()
        data['nac'] = np.concatenate([data['actions'][1:], data['end_ac']], axis=0)
        data['nobs'] = np.concatenate([data['obs'][1:], data['end_ob']], axis=0)
        return data

    def _update_model(self, data):
        if self.rn is not None:
            self._update_running_norm(data['obs'].reshape(-1, *data['obs'].shape[2:]))
        dataset, _ = util.make_dataset(data)
        batch = dataset.data_map
        for _ in range(self.args.epochs):
            losses = []
            for b in dataset.iterate_once(self.batch_size * self.nenv):
                out = self._update(b)
                losses.append([out[k] for k in self._loss_keys])
            meanlosses = np.array(losses).mean(axis=0)
            s = 'Losses:  '
            for i,ln in enumerate(self._loss_names):
                s += ln + ': {:08f}  '.format(meanlosses[i])
            logger.log(s)
        return meanlosses

    def _update(self, batch):
        inputs=[batch['obs'], batch['actions'], batch['nobs'], batch['nac'], batch['rewards'], batch['dones']]
        loss_kwargs = {l:True for l in self._loss_keys}
        return self.opt.run(inputs=inputs, state=[], state_out=False, grad=False, update=True, **loss_kwargs)

    def _after_step(self, rollout, data, outs):
        logger.log("========================|  Timestep: {}  |========================".format(self.t))
        logger.logkv('serial timesteps', self.t / self.nenv)
        logger.logkv('mean episode length', np.mean(self.runner.get_episode_lengths()))
        logger.logkv('mean episode reward', np.mean(self.runner.get_episode_rewards()))
        logger.logkv('fps', int((self.t - self._t_start) / (time.time() - self._time_start)))
        logger.logkv('time_elapsed', time.time() - self._time_start)
        logger.logkv('time spent exploring', self.actor.eps)
        logger.dumpkvs()

    def _update_running_norm(self, x):
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        count = x.shape[0]
        self.rn.update(mean, var, count)

    def update_lr(self, new_lr):
        self.opt.update_lr(new_lr)
