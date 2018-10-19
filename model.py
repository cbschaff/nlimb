import tensorflow as tf
import numpy as np
from deeplearning import tf_util as U, layers, module
from deeplearning.distributions import DiagGaussianMixturePd
from rl.rl_module import Policy, ActorCritic, ValueFunction

class GmmPd(DiagGaussianMixturePd):
    """
    Change GMM model to only allow gradients
    through the sampled component.
    """
    def sample(self):
        samples = tf.stack([g.sample() for g in self.gaussians])
        m = self.mixture.sample()
        s = tf.concat([tf.gather(samples, m)[0], tf.cast(m, tf.float32)[None]], axis=1)
        return s

    def mode(self):
        modes = tf.stack([g.mode() for g in self.gaussians])
        logps = tf.stack([g.logp(g.mode()) + self.log_mixing_probs[:,i] for i,g in enumerate(self.gaussians)])
        m = tf.argmax(logps)
        s = tf.concat([tf.gather(modes, tf.argmax(logps))[0], tf.cast(m, tf.float32)[None]], axis=1)
        return s

    def neglogp(self, x):
        params = x[:,:-1]
        comp = tf.cast(x[:,-1:], tf.int32)
        comp = tf.concat([comp, tf.expand_dims(tf.range(comp.shape[0]),axis=1)], axis=1)
        p = tf.stack([self.log_mixing_probs[:,i] + self.gaussians[i].logp(params) for i in range(self.n)])
        p = tf.gather_nd(p, comp)
        return -1. * p


class Net(module.Module):
    """
    Fully connected network.
    """
    ninputs=1
    def __init__(self, name, *modules, hiddens=[], activation_fn=tf.nn.tanh):
        super().__init__(name, *modules)
        self.hiddens = hiddens
        self.activation_fn = activation_fn

    def _build(self, inputs):
        net = tf.clip_by_value(inputs[0], -5.0, 5.0)
        for i,h in enumerate(self.hiddens):
            net = tf.layers.dense(
                net,
                units=h,
                kernel_initializer=U.normc_initializer(1.0),
                activation=self.activation_fn,
                name='dense{}'.format(i)
            )
        return net

class RobotSampler(module.Module):
    """
    Define robot distribution.
    """
    ninputs=1
    def __init__(self, name, robot, nparams, ncomponents=8, mean_init=None, std_init=0.577):
        super().__init__(name, robot)
        self.nparams = nparams
        self.ncomponents = ncomponents
        self.mean_init = mean_init
        self.std_init = std_init

    def _build(self, inputs):
        sampled_robot = inputs[0]
        vars = []
        vars.append(tf.get_variable('mixprobs',
                                     shape=(self.ncomponents,),
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     trainable=False))
        for i in range(self.ncomponents):
            if self.mean_init is not None:
                mean = np.asarray(self.mean_init)
            else:
                mean = np.random.uniform(-0.8,0.8, size=self.nparams)
            m = tf.get_variable('m{}'.format(i),
                                shape=mean.shape,
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(mean))
            logstd = np.log(self.std_init * np.ones_like(mean))
            s = tf.get_variable('logstd{}'.format(i),
                                shape=logstd.shape,
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(logstd))
            vars.append(m)
            vars.append(s)
        gmm_params = tf.tile(tf.expand_dims(tf.concat(vars, axis=0), axis=0), [self.nbatch*self.nstep, 1])
        self.pd = GmmPd(gmm_params, self.ncomponents)


        self._sample_component = self.pd.mixture.sample()
        self._sample_gaussians = [g.sample() for g in self.pd.gaussians]
        self._mode = self.pd.mode()
        self._sample = self.pd.sample()
        return self.pd.neglogp(sampled_robot)

    def sample(self, stochastic=True):
        if not stochastic:
            return self._mode.eval()
        else:
            return self._sample.eval()

    def sample_component(self):
        return self._sample_component.eval()

    def sample_gaussian(self, index):
        s = self._sample_gaussians[index].eval()
        return np.concatenate([s, [[index]]], axis=1)

class RunningObsNorm(layers.RunningNorm):
    """
    Only normalize observations, not robot params.
    """
    ninputs=1
    def __init__(self, name, *modules, param_size=None):
        assert param_size is not None
        self.size = param_size
        super().__init__(name, *modules)

    def _build(self, inputs):
        X = inputs[0]
        obs = X[:,:-self.size]
        obs_normed = super()._build([obs])
        return tf.concat([obs_normed, X[:,-self.size:]], axis=-1)

    def update(self, mean, var, count):
        super().update(mean[:-self.size], var[:-self.size], count)

class Model(ActorCritic):
    """
    Combine policy, value function and robot distribution in one Module.
    """
    def __init__(self, name, policy, value_function, robot_sampler):
        super().__init__(name, policy, value_function)
        self.sampler = robot_sampler
