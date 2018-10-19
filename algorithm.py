from deeplearning import tf_util as U, logger, module, layers
from deeplearning.mpi import MpiAdam, mpi_moments
from rl.algorithms import MPIPPO
import sys, os, shutil, glob, json, time
import tensorflow as tf
import numpy as np
from component_chopper import ComponentChopper


class FlatGrad(module.Module):
    ninputs=1
    def __init__(self, name, *modules, clip_norm=None):
        super().__init__(name, *modules)
        self.clip_norm = clip_norm

    def _build(self, inputs):
        loss = inputs[0]
        params = self.trainable_variables()
        return U.flatgrad(loss, params, self.clip_norm)


class RobotDistributionLoss(module.Module):
    def _build(self, inputs):
        neglogp, episode_reward = inputs
        assert neglogp.shape == episode_reward.shape
        return tf.reduce_mean(neglogp * episode_reward)


class Algorithm(MPIPPO):
    """
    Extends the PPO algorithm to:
    1) sample a new robot for each iteration
    2) update the robot distribution to maximize reward under the current policy.
    """
    def defaults(self):
        defaults = super().defaults()
        defaults.update({
            'robot_lr':1e-3,
            'robot_momentum':0.9,
            'fixed_robot':False,
            'steps_before_robot_update':int(1e8),
            'steps_after_robot_update':int(1e8),
            'chop_freq':int(1e8),
            'tmax':int(1e9)
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(os.path.join(self.logdir, 'summaries'), exist_ok=True)
        self.writer = tf.summary.FileWriter(os.path.join(self.logdir, 'summaries'), max_queue=10000, flush_secs=60)
        self.chopper = ComponentChopper(self.env, self.actor, self.mpi_rank)

    def _build_robot_dist_optimizer(self):
        r = layers.Placeholder(tf.float32, [], 'robot_reward')
        robot_loss = RobotDistributionLoss('rloss', self.actor.sampler, r)
        self.robot_grad = FlatGrad('rgrad', robot_loss, clip_norm=self.args.clip_norm)
        self.robot_grad.build('model', 1, 1)
        self.mpi_adam_robot = MpiAdam(self.robot_grad.trainable_variables(), epsilon=1e-5, beta1=self.args.robot_momentum, beta2=self.args.beta2)

    def _def_loss(self, model_fn, env):
        self._build_robot_dist_optimizer()
        return super()._def_loss(model_fn, env)

    def load(self, t=None):
        super().load(t)
        path = os.path.join(self.logdir, 'checkpoints', str(self.t), 'radam.npz')
        if hasattr(self, 'mpi_adam_robot') and os.path.exists(path):
            self.mpi_adam_robot.load(path)

        # load robot xml file.
        xml = os.path.join(self.logdir, 'checkpoints', str(self.t), 'design.xml')
        if os.path.exists(xml):
            shutil.copyfile(xml, self.env.unwrapped.model_xml)

    def save(self):
        super().save()
        if self.mpi_rank == 0:
            self.mpi_adam_robot.save(os.path.join(self.logdir, 'checkpoints', str(self.t), 'radam.npz'))

            # save mode robot xml
            self.sample_robot(stochastic=False)
            xml = os.path.join(self.logdir,'checkpoints', str(self.t), 'design.xml')
            shutil.copyfile(self.env.unwrapped.model_xml, xml)

    def sync(self):
        super().sync()
        self.mpi_adam_robot.sync()

    def sample_robot(self, stochastic=True):
        robot = self.actor.sampler.sample(stochastic=stochastic)[0]
        self.env.update_robot(robot)
        self.runner.reset()

    def _before_step(self):
        super()._before_step()
        # decay learning rate
        self._lr_frac = max(0.0, 1.0 - self.t / self.args.tmax)
        super().update_lr(self.args.lr * self._lr_frac)


        # chop GMM components
        t_prev = self.t - self.timesteps_per_step
        last_chop = t_prev // self.args.chop_freq
        if self.t // self.args.chop_freq != last_chop and t_prev > self.args.steps_before_robot_update:
            if self.chopper.components_left() > 1:
                self.chopper.chop()
                self.mpi_adam.reset()
                self.mpi_adam_robot.reset()

        stochastic = self.args.tmax - self.t >= self.args.steps_after_robot_update and not self.args.fixed_robot
        self.sample_robot(stochastic)

    def _update_model(self, data):
        losses = super()._update_model(data)
        if self.t >= self.args.steps_before_robot_update and self.args.tmax - self.t >= self.args.steps_after_robot_update and not self.args.fixed_robot:
            self._update_robot_dist()
        return losses

    def _update_robot_dist(self):
        self.env.update_buffer()
        episode_reward = [self.env.reward_buffer[-1]]
        params = [self.env.param_buffer[-1]]
        r = self._norm_rewards(episode_reward)
        grad = self.robot_grad([params, r])
        self.mpi_adam_robot.update(grad, self.args.robot_lr * self._lr_frac)

    def _norm_rewards(self, rewards):
        mean, std, _ = mpi_moments(rewards)
        return (rewards - mean) / (std + 1e-8)

    def _after_step(self, *args):
        super()._after_step(*args)
        if self.mpi_rank == 0:
            self.write_summary()

    def write_summary(self):
        summary = tf.Summary()
        avg_length = np.mean(self.runner.get_episode_lengths())
        avg_reward = np.mean(self.runner.get_episode_rewards())
        summary.value.add(tag="episode/length", simple_value=float(avg_length))
        summary.value.add(tag="episode/reward", simple_value=float(avg_reward))

        for name,param in zip(self.env.param_names, self.env.robot.get_params()):
            summary.value.add(tag="robot/" + name, simple_value=float(param))
            summary.value.add(tag="robot/" + name, simple_value=float(param))
        self.writer.add_summary(summary, global_step=self.t)
