import numpy as np
import tensorflow as tf
from rl.losses import QLearningLoss
from rl.algorithms import OnlineRLAlgorithm
from rl.runner import *
from rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rl import util
from deeplearning.layers import Adam, RunningNorm
from deeplearning.schedules import LinearSchedule
from deeplearning import logger
from collections import deque
import time

class QLearning(OnlineRLAlgorithm):
    def defaults(self):
        return {
            'lr': 1e-4,
            'momentum': 0.9,
            'beta2': 0.999,
            'clip_norm': 10.,
            'gamma': 0.99,
            'learning_starts': int(1e5),
            'exploration_timesteps': int(1e6),
            'final_eps': 0.02,
            'target_update_freq': int(1e4),
            'prioritized_replay': True,
            'huber_loss': True,
            'buffer_size': int(1e6),
            'replay_alpha': 0.6,
            'replay_beta': 0.4,
            't_beta_max': int(1e7)
        }

    def __init__(self,
        logdir,
        env_fn,
        model_fn,
        nenv,
        rollout_length=1,
        batch_size=32,
        callback=None,
        **kwargs
    ):
        defaults = self.defaults()
        for k in kwargs:
            assert k in defaults, "Unknown argument: {}".format(k)
        defaults.update(kwargs)

        super().__init__(logdir, env_fn, model_fn, nenv, rollout_length, batch_size, callback, runner_flags=[], **defaults)

        self.target_sync = tf.group([tf.assign(v1,v2) for v1,v2 in zip(self.loss.qtarg.variables(), self.loss.qvals.variables())])
        if self.args.prioritized_replay:
            self.buffer = PrioritizedReplayBuffer(self.args.buffer_size, alpha=self.args.replay_alpha)
        else:
            self.buffer = ReplayBuffer(self.args.buffer_size)
        # determine if the network has a RunningNorm submodule that needs to be updated.
        submods = self.opt.find_submodules_by_instance(RunningNorm)
        self.rn = submods[0] if len(submods) > 0 else None

        self.losses = deque(maxlen=100)
        self.nsteps = 0
        self.last_target_sync = (self.t // self.args.target_update_freq) * self.args.target_update_freq
        self.beta_schedule = LinearSchedule(self.args.t_beta_max, 1.0, self.args.replay_beta)
        self.eps_schedule = LinearSchedule(int(self.args.exploration_timesteps), self.args.final_eps, 1.0)
        self._time_start = time.time()
        self._t_start = self.t

    def _def_loss(self, model_fn, env):
        target_network = model_fn(env)
        target_network.build('target', self.nenv, self.batch_size, trainable=False)
        # extra network for double dqn. Tie variables with network
        return QLearningLoss('loss', model_fn(env), model_fn(env), target_network, gamma=self.args.gamma, use_huber_loss=self.args.huber_loss)

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
        if self.t == 0 or self.t - self.last_target_sync > self.args.target_update_freq:
            self.target_sync.run()
            self.last_target_sync = self.t
        self.actor.update_eps(self.eps_schedule.value(self.t))

    def _process_rollout(self, rollout):
        self._update_buffer(rollout)
        while len(self.buffer) < self.args.learning_starts and len(self.buffer) != self.args.buffer_size:
            self._update_buffer(self.runner.rollout())
            self.t += self.timesteps_per_step

        if self.args.prioritized_replay:
            obs, acs, rews, next_obs, dones, weights, self._inds = self.buffer.sample(self.nenv * self.batch_size, self.beta_schedule.value(self.t))
            inputs=[obs, next_obs, next_obs, rews, acs, dones, weights[...,None]]
        else:
            obs, acs, rews, next_obs, dones = self.buffer.sample(self.nenv * self.batch_size)
            inputs=[obs, next_obs, next_obs, rews, acs, dones]
        return inputs

    def _update_buffer(self, rollout):
        if self.rn is not None:
            x = np.asarray(rollout.obs)
            self._update_running_norm(x.reshape([-1] + list(x.shape[2:])))
        for i,obs in enumerate(rollout.obs):
            next_obs = rollout.end_ob if i == len(rollout.obs) - 1 else rollout.obs[i+1]
            for j in range(self.nenv):
                ob = obs[j]
                next_ob = next_obs[j]
                ac = rollout.actions[i][j]
                r = rollout.rewards[i][j]
                done = rollout.dones[i][j]
                self.buffer.add(ob, ac, r, next_ob, done)

    def _update_model(self, data):
        outs = self.opt.run(inputs=data, state=[], state_out=False, update=True, td=True)
        if self.args.prioritized_replay:
            self.buffer.update_priorities(self._inds, priorities=np.abs(outs['td'][:,0]) + 1e-6)
        self.losses.append(outs['out'])
        return outs

    def _after_step(self, rollout, data, outs):
        self.nsteps += 1
        if self.nsteps % 100 == 0:
            logger.log("========================|  Timestep: {}  |========================".format(self.t))
            meanloss = np.mean(np.array(self.losses), axis=0)
            # Logging stats...
            logger.logkv('Loss', meanloss)
            logger.logkv('timesteps', self.t)
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
