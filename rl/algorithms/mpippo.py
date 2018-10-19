import numpy as np
import tensorflow as tf
from rl.algorithms import PPO
from rl.runner import *
from rl import util
from deeplearning.layers import Adam
from deeplearning import logger
import time, sys, os
from deeplearning.mpi import MpiAdam, mpi_mean, mpi_moments, mpi_fork
from mpi4py import MPI

class MPIPPO(PPO):
    def __init__(self, *args, **kwargs):
        self.mpi_rank = MPI.COMM_WORLD.Get_rank()
        super().__init__(*args, **kwargs)
        if self.mpi_rank > 0:
            logger.set_level(logger.DISABLED)
        self.mpi_adam = MpiAdam(self.loss.trainable_variables(), epsilon=1e-5, beta1=self.args.momentum, beta2=self.args.beta2)
        self.load()
        self.sync()
        self.timesteps_per_step = self.rollout_length * MPI.COMM_WORLD.Get_size()

    def _make_env(self, env_fn, nenv):
        whoami = mpi_fork(nenv)
        if whoami == "parent": sys.exit(0)
        self.nenv = 1
        return env_fn(self.mpi_rank)

    def sync(self):
        self.mpi_adam.sync()

    def _update(self, batch, state):
        if self.loss.is_recurrent:
            inputs=[batch['obs'], batch['dones'], batch['actions'], batch['vtarg'], batch['atarg'], batch['neglogp'], batch['vpreds']]
        else:
            inputs=[batch['obs'], batch['actions'], batch['vtarg'], batch['atarg'], batch['neglogp'], batch['vpreds']]
        # update the network and return losses
        outs = self.opt.run(inputs=inputs, state=state, flatgrad=True, p_loss=True, v_loss=True, ent_loss=True)
        grad = outs['flatgrad']
        self.mpi_adam.update(outs['flatgrad'], self.opt.lr)
        losses = [[outs['out'], outs['p_loss'], outs['v_loss'], outs['ent_loss']]]
        ml, _ = mpi_mean(losses, axis=0)
        outs['out'] = ml[0]
        outs['p_loss'] = ml[1]
        outs['v_loss'] = ml[2]
        outs['ent_loss'] = ml[3]
        return outs

    # def _norm_advantages(self, advs):
    #     mean, std, _ = mpi_moments(advs)
    #     return (advs - mean) / (std + 1e-8)

    def _update_running_norm(self, x):
        mean, std, count = mpi_moments(x.astype(np.float32), axis=0)
        self.rn.update(mean, std*std, count)

    def _after_step(self, rollout, data, meanlosses):
        for i,s in enumerate(['Total Loss', 'Policy Loss', 'Value Loss', 'Entropy']):
            logger.logkv(s, meanlosses[i])
        vtarg_flat = data['vtarg'].flatten()
        vpred_flat = data['vpreds'].flatten()

        logger.logkv('timesteps', self.t)
        logger.logkv('serial timesteps', self.t / self.nenv)
        ep_len, _ = mpi_mean(self.runner.get_episode_lengths())
        logger.logkv('mean episode length', ep_len)
        ep_rew, _ = mpi_mean(self.runner.get_episode_rewards())
        logger.logkv('mean episode reward', ep_rew)
        logger.logkv('explained var. of vtarg', util.explained_variance(vpred_flat, vtarg_flat))
        logger.logkv('fps', int((self.t - self._t_start) / (time.time() - self._time_start)))
        logger.logkv('time_elapsed', time.time() - self._time_start)
        logger.dumpkvs()

    def save(self):
        if self.mpi_rank == 0:
            super().save()
            self.mpi_adam.save(os.path.join(self.logdir, 'checkpoints', str(self.t), 'adam.npz'))

    def load(self, t=None):
        super().load(t)
        path = os.path.join(self.logdir, 'checkpoints', str(self.t), 'adam.npz')
        if hasattr(self, 'mpi_adam') and os.path.exists(path):
            self.mpi_adam.load(path)



if __name__ == '__main__':
    from deeplearning import tf_util as U
    from rl.test import model
    import shutil

    U.reset()
    def env_fn(rank):
        return util.make_single_threaded_atari_env('PongNoFrameskip-v4', 0, rank, {'frame_stack':True})

    ppo = MPIPPO('./logs/ppo', env_fn, model, 2, epochs_per_iter=2, rollout_length=128)
    ppo.train(maxtimesteps=1024)
    if ppo.mpi_rank == 0:
        shutil.rmtree('./logs')
