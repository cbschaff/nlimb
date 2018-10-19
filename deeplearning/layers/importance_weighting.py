import tensorflow as tf
from deeplearning import module
from deeplearning import tf_util as U

class GradientImportanceWeighting(module.Module):
    """
    This module takes three inputs:
        1) A module whose output op is to have its gradients reweigthed.
        2) A module which outputs neglogps
        3) A a module or placeholder for sample distrubition's neglogps
    """
    ninputs = 3
    def _build(self, inputs):
        op, neglogp, sample_dist_neglogp = inputs
        self.iw = tf.exp(sample_dist_neglogp - neglogp)
        while len(self.iw.shape) < len(op.shape):
            self.iw = tf.expand_dims(self.iw, -1)
        @tf.custom_gradient
        def iw(x):
            def grad(dy):
                return self.iw * dy
            return x, grad
        return iw(op)

    def _add_run_args(self, outs, feed_dict, **flags):
        super()._add_run_args(outs, feed_dict, **flags)
        if 'iw' in flags and flags['iw']:
            outs['iw'] = self.iw









# Test
if __name__=='__main__':
    from deeplearning import distributions as dist
    from deeplearning import layers
    import numpy as np

    class DiagGaussian(module.Module):
        ninputs=2 # pdparams, sample
        def _build(self, inputs):
            pdparams, sample = inputs
            gaussian = dist.DiagGaussianPd(pdparams)
            return gaussian.neglogp(sample)



    X = layers.Placeholder(tf.float32, [1], 'input')
    ph1 = layers.Placeholder(tf.float32, [2], 'params1')
    ph2 = layers.Placeholder(tf.float32, [2], 'params2')
    net = layers.Dense('out', X, units=1)
    pdist1 = DiagGaussian('dist', ph1, X)
    pdist2 = DiagGaussian('sample_dist', ph2, X)
    net = GradientImportanceWeighting('iw', net, pdist1, pdist2)
    opt = layers.SGD('opt', net)

    opt.build('model', 2, 1)

    p1 = np.array([[0,0],[0,0]])
    p2 = np.array([[1,0],[1,0]])
    x = np.array([[-1],[1]])

    with U.single_threaded_session() as sess:
        U.initialize()
        outs = opt.run(inputs=[x, p1, p2], grad=True, update=True, iw=True)
        print(x)
        print(outs['grad'])
        print(outs['out'])
        print(outs['iw'])
        print(opt.run(inputs=[x, p1, p2]))
