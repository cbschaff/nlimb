"""
Adapted from OpenAI Baselines.
"""

import os, random
import gym
import numpy as np
from scipy.signal import lfilter
from deeplearning import logger
from deeplearning.dataset import Dataset
from rl.monitor import Monitor
from rl.atari_wrappers import make_atari, wrap_deepmind
from rl.vec_env.subproc_vec_env import SubprocVecEnv

"""
Math Util
"""

def discount(x, gamma):
    return lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def gae(data, gamma, lambda_=1.0):
    """
    Process a rollout using the formula from:
    "Generalized Advantage Estimation"
    https://arxiv.org/abs/1506.02438
    """
    T = len(data['rewards'])
    shape = list(data['rewards'][0].shape)
    data['atarg'] = np.empty([T] + shape, dtype=np.float32)
    data['vtarg'] = np.empty([T] + shape, dtype=np.float32)
    vpred = np.concatenate((data['vpreds'], data['end_vpred']), axis=0)
    if len(vpred.shape) > len(data['dones'].shape):
        nonterminal = (1.0 - data['dones'])[...,None]
    else:
        nonterminal = (1.0 - data['dones'])
    dt = data['rewards'] + gamma * nonterminal * vpred[1:] - vpred[:-1]
    adv_t_plus_1 = np.zeros(shape)
    for t in reversed(range(T)):
        data['atarg'][t] = adv_t_plus_1 = dt[t] + gamma * lambda_ * nonterminal[t] * adv_t_plus_1
    data['vtarg'] = data['atarg'] + data['vpreds']

def make_dataset(data, recurrent=False):
    return Dataset({k:flatten01(v) for k,v in data.items() if k not in ['state', 'end_ob', 'end_vpred', 'end_ac']}, deterministic=recurrent), data['state']

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def explained_variance_2d(ypred, y):
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y-ypred)/vary
    out[vary < 1e-10] = 0
    return out

"""
Numpy Util.
"""
def flatten01(x):
    return x.reshape(-1, *x.shape[2:])

"""
Random Seeds
"""
def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

"""
OpenAI Environments
"""
def make_atari_env(env_id, num_env, seed, wrapper_kwargs={}, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    def make_env(rank): # pylint: disable=C0111
        def _make_env():
            return make_single_threaded_atari_env(env_id, seed, rank, wrapper_kwargs)
        return _make_env
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_single_threaded_atari_env(env_id, seed, rank=0, wrapper_kwargs={}):
    env = make_atari(env_id)
    env.seed(seed + rank)
    env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    return wrap_deepmind(env, **wrapper_kwargs)

def make_mujoco_env(env_id, seed):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = Monitor(env, logger.get_dir())
    env.seed(seed)
    return env
