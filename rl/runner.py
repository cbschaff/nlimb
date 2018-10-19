from rl.vec_env import VecEnv, VecEnvWrapper
from rl.vec_env.dummy_vec_env import DummyVecEnv
from rl import util
from deeplearning.dataset import Dataset
import tensorflow as tf
import threading
import six.moves.queue as queue
import numpy as np
from collections import deque


# Stores rollout data from interacting with an environment.
class Rollout(object):
    def __init__(self, initial_state, *flags):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.vpreds = []
        self.dones = []
        self.info = []
        self.state = initial_state
        self.end_vpred = None
        self.end_ob = None
        self.end_ac = None
        self.flags = flags
        for f in flags:
            setattr(self, f, [])

    def add(self, ob, action, reward, done, vpred=None, info=None, **flag_data):
        self.obs.append(ob)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.vpreds.append(vpred)
        self.info.append(info)
        for f in self.flags:
            setattr(self, f, getattr(self, f) + [flag_data[f]])

    def extend(self, other):
        self.obs.extend(other.obs)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)
        self.vpreds.extend(other.vpreds)
        self.info.extend(other.info)
        self.end_vpred = other.end_vpred
        self.end_ob = other.end_ob
        self.end_ac = other.end_ac
        for f in self.flags:
            assert f in other.flags
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def numpy(self):
        data = {}
        data['obs'] = np.asarray(self.obs)
        data['actions'] = np.asarray(self.actions)
        data['rewards'] = np.asarray(self.rewards)
        data['dones'] = np.asarray(self.dones)
        if len(self.vpreds) == 0 or self.vpreds[-1] is not None:
            data['vpreds'] = np.asarray(self.vpreds)
        if self.end_vpred is not None:
            data['end_vpred'] = np.asarray(self.end_vpred)[None]
        data['end_ob'] = np.asarray(self.end_ob)[None]
        data['end_ac'] = np.asarray(self.end_ac)[None]
        for f in self.flags:
            data[f] = np.asarray(getattr(self, f))
        data['state'] = self.state
        return data


# Samples rollouts from an environment with a given actor.
class Runner(object):
    """
    A Runner uses an actor (i.e. policy, qfunction, ...) to interact with an environment.

    Inputs:
        - env: a OpenAI Gym Environment.
        - actor: An ActortCritic Module.
        - rollout_length: The number of timesteps to run in the environment before returning a Rollout.
    """
    def __init__(self, env, actor, rollout_length, *flags):
        # vectorize env if needed.
        if isinstance(env, VecEnv) or isinstance(env, VecEnvWrapper):
            self.env = env
        else:
            self.env = DummyVecEnv([lambda: env])
        self.actor = actor
        self.rollout_length = rollout_length
        self.ob = self.env.reset()
        self.state = []
        self._episode_rewards = deque(maxlen=100)
        self._episode_lengths = deque(maxlen=100)
        self.rew = np.zeros([self.env.num_envs])
        self.len = np.zeros([self.env.num_envs])
        self.done = np.zeros([self.env.num_envs], dtype=np.bool)
        self.flags = flags
        self.args = {f:True for f in flags}


    def reset(self):
        self.ob = self.env.reset()
        self.rew = np.zeros([self.env.num_envs])
        self.len = np.zeros([self.env.num_envs])
        self.state = []
        self.done = np.zeros([self.env.num_envs], dtype=np.bool)

    def step(self):
        if self.actor.is_recurrent:
            outs = self.actor.run([self.ob, self.done], self.state, **self.args)
        else:
            outs = self.actor.run(self.ob, self.state, **self.args)
        out = outs['out']
        self.state = outs['state_out']
        if isinstance(out, list) or isinstance(out, tuple):
            action = out[0]
            value = out[1]
            # expand dim for consistency
            if len(value.shape) == 1:
                value = value[:,None]
        else:
            action = out
            value = None
        self.ob, reward, done, info = self.env.step(action)
        self.done = done.copy() # save this for recurrent models.
        # keep track of recent episodes.
        self.rew += reward.reshape(self.rew.shape)
        self.len += 1
        for i,d in enumerate(done):
            if d:
                self._episode_rewards.append(self.rew[i])
                self._episode_lengths.append(self.len[i])
                self.rew[i] = 0
                self.len[i] = 0
        # expand dim for consistency
        if len(reward.shape) == 1:
            reward = reward[:,None]
        # if len(action.shape) == 1:
        #     action = action[:,None]
        return self.ob, action, reward, done, value, info, {k:outs[k] for k in outs if k in self.flags}

    def rollout(self):
        rollout = Rollout(self.state, *self.flags)
        for i in range(self.rollout_length):
            prev_ob = self.ob
            ob, action, reward, done, value, info, flag_data = self.step()
            rollout.add(prev_ob, action, reward, done, value, info, **flag_data)
        rollout.end_ob = self.ob
        if self.actor.is_recurrent:
            outs = self.actor.run([self.ob, self.done], self.state)
        else:
            outs = self.actor.run(self.ob, self.state)
        out = outs['out']
        if rollout.vpreds[-1] is not None:
            rollout.end_ac = out[0]
            rollout.end_vpred = out[1]
        else:
            rollout.end_ac = out
        return rollout

    def get_episode_lengths(self):
        return self._episode_lengths

    def get_episode_rewards(self):
        return self._episode_rewards

# Launches a runner in its own thread.
class RunnerThread(threading.Thread):
    def __init__(self, runner):
        threading.Thread.__init__(self)
        self.runner = runner
        self.daemon = True
        self._stop_event = threading.Event()

    def start(self):
        self.sess = tf.get_default_session()
        self.queue = queue.Queue(5)
        super().start()

    def stop(self):
        self._stop_event.set()
        self.join()

    def should_stop(self):
        return self._stop_event.isSet()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout = self.runner.rollout()
        while not self.should_stop():
            try:
                self.queue.put(rollout, timeout=2.0)
                rollout = self.runner.rollout()
            except:
                pass

    def rollout(self):
        try:
            return self.queue.get(timeout=10.0)
        except:
            assert False, "Could not fetch batch from runner!!"
            self.stop()

    def get_episode_lengths(self):
        return self.runner._episode_lengths

    def get_episode_rewards(self):
        return self.runner._episode_rewards
