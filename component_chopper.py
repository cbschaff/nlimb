import numpy as np
from mpi4py import MPI


class ComponentChopper(object):
    def __init__(self, env, actor, rank):
        self.env = env
        self.actor = actor
        self.sampler = actor.sampler
        vs = [v for v in self.sampler.variables() if 'mixprobs' in v.name]
        assert len(vs) == 1, "Should only be one mixprobs variable. Found %d"%len(vs)
        self.tflogmixprobs = vs[0]
        self.logmixprobs = self.tflogmixprobs.eval()
        self.rank = rank

    def components_left(self):
        return np.sum(self.logmixprobs == 0.)

    def test_robot(self, robot):
        reward = 0.
        done = False
        self.env.update_robot(robot)
        ob = self.env.reset()
        while not done:
            ac = self.actor.act(ob[None])[0]
            ob, rew, done, _ = self.env.step(ac)
            reward += rew
        return reward

    def chop(self):
        n = self.components_left()
        if n <= 1: return

        if self.rank == 0:
            avg_component_reward = np.zeros_like(self.logmixprobs)
            for i,c in enumerate(self.logmixprobs):
                if c != 0: continue
                rews = []
                for _ in range(100):
                    robot = self.sampler.sample_gaussian(i)[0]
                    rews.append(self.test_robot(robot))
                avg_component_reward[i] = np.mean(rews)
            inds = np.argsort(avg_component_reward)
            for ind in inds:
                if self.components_left() <= n // 2:
                    break
                self.logmixprobs[ind] = -1000000.

            self.tflogmixprobs.load(self.logmixprobs)
        self.sync()

    def sync(self):
        if self.rank == 0:
            MPI.COMM_WORLD.Bcast(self.tflogmixprobs.eval(), root=0)
        else:
            MPI.COMM_WORLD.Bcast(self.logmixprobs, root=0)
            self.tflogmixprobs.load(self.logmixprobs)
