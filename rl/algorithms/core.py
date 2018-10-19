from deeplearning import logger, tf_util as U
import tensorflow as tf
from rl.runner import Runner
from rl.vec_env.subproc_vec_env import SubprocVecEnv
from collections import namedtuple
import os, time

class RLExperiment(U.Experiment):
    def load_env_fn(self):
        fname = os.path.join(self.logdir, 'checkpoints/env_fn.pkl')
        assert os.path.exists(fname), "No env function saved."
        return U.load(fname)

    def save_env_fn(self, env_fn):
        fname = os.path.join(self.logdir, 'checkpoints/env_fn.pkl')
        U.save(fname, env_fn)



class OnlineRLAlgorithm(object):
    def __init__(self, logdir, env_fn, model_fn, nenv, rollout_length, batch_size, callback=None, runner_flags=[], **kwargs):
        self.exp = RLExperiment(logdir)
        self.exp.save_model_fn(model_fn)
        self.exp.save_env_fn(env_fn)
        logger.configure(os.path.join(logdir, 'logs'), ['stdout', 'log', 'json'])
        self.logdir = logdir
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.args = namedtuple('Args', kwargs.keys())(**kwargs)

        self.nenv = nenv
        self.timesteps_per_step = self.nenv * self.rollout_length
        self.env = self._make_env(env_fn, nenv)

        self.actor = model_fn(self.env)
        self.actor.build('model', self.nenv, 1)

        self.loss = self._def_loss(model_fn, self.env)
        self.opt = self._def_opt(self.loss)
        self.opt.build('model', self.nenv, batch_size, reuse=tf.AUTO_REUSE)

        self.runner = Runner(self.env, self.actor, rollout_length, *runner_flags)

        self.callback = callback
        if callback is not None:
            assert callable(callback)

        self.init_session()
        self.load()

    def _make_env(self, env_fn, nenv):
        def make_env(rank):
            def _env():
                return env_fn(rank)
            return _env
        return SubprocVecEnv([make_env(i) for i in range(nenv)])

    def _def_loss(self, model_fn, env):
        """
        returns a module for and the loss
        """
        raise NotImplementedError

    def _def_opt(self, loss):
        """
        returns a module for and the optimizer
        """
        raise NotImplementedError

    def _before_step(self):
        pass

    def _process_rollout(self, rollout):
        raise NotImplementedError

    def _update_model(self, data):
        raise NotImplementedError

    def _after_step(self, rollout, data, update_out):
        pass

    def step(self):
        if self.callback is not None:
            self.callback(locals(), globals())
        self._before_step()
        rollout = self.runner.rollout()
        self.t += self.timesteps_per_step
        data = self._process_rollout(rollout)
        outs = self._update_model(data)
        self._after_step(rollout, data, outs)

    def train(self, maxtimesteps=None, maxseconds=None, save_freq=None):
        assert maxtimesteps is not None or maxseconds is not None
        start_time = time.time()
        while True:
            if maxtimesteps is not None and self.t >= maxtimesteps:
                break
            if maxseconds is not None and time.time() - start_time >= maxtimesteps:
                break
            t = self.t
            self.step()
            if save_freq and t // save_freq != self.t // save_freq:
                self.save()
        self.save()

    def save(self):
        self.exp.save(self.t)

    def load(self, t=None):
        self.t = self.exp.load(t)

    def init_session(self):
        if tf.get_default_session() is None:
            U.make_session().__enter__()
        U.initialize()

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
        tf.get_default_session().__exit__(None, None, None)
        logger.reset()




if __name__=='__main__':
    from deeplearning.layers import Adam, Placeholder
    from deeplearning.module import Module
    from rl.rl_module import Policy
    import tensorflow as tf
    import gym
    from rl import util

    class TestAlg(OnlineRLAlgorithm):
        def _def_loss(self, model_fn):
            class Ent(Module):
                def _build(self, inputs):
                    return self.modules[0]._entropy
            return Ent('l', model_fn(self.env))

        def _def_opt(self, loss):
            return Adam('opt', loss)

        def _before_step(self):
            logger.log("Before Step")

        def _process_rollout(self, rollout):
            return rollout.numpy()

        def _update_model(self, data):
            self.opt.update(util.swap01andflatten(data['obs']))

        def _after_step(self, rollout, data, update_outs):
            logger.log("After Step")

    def model_fn(env):
        x = Placeholder(tf.float32, env.observation_space.shape, 'x')
        return Policy('pi', x, ac_space=env.action_space)

    def env_fn(rank):
        env = gym.make('CartPole-v1')
        env.seed(rank)
        return env

    alg = TestAlg('./test_logs', env_fn, model_fn, 2, 64, 64)

    alg.train(1024, save_freq=128)
