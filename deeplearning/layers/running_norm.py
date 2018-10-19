import tensorflow as tf
from deeplearning import module
from deeplearning import tf_util as U

class RunningNorm(module.Module):
    ninputs=1
    def _build(self, inputs):
        X = inputs[0]
        shape = X.shape[1:]
        self._count = tf.get_variable('running_norm_count', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
        self._mean = tf.get_variable('running_norm_mean', shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
        self._var = tf.get_variable('running_norm_var', shape=shape, dtype=tf.float32, initializer=tf.ones_initializer(), trainable=False)

        self._batch_mean = tf.placeholder(X.dtype, shape=shape, name='batch_mean_ph')
        self._batch_var = tf.placeholder(X.dtype, shape=shape, name='batch_var_ph')
        self._batch_count = tf.placeholder(X.dtype, shape=(), name='batch_count_ph')
        delta = self._batch_mean - self._mean
        new_count = self._count + self._batch_count
        new_mean = self._mean + delta * (self._batch_count / new_count)
        new_var = self._count * self._var + self._batch_count * self._batch_var
        new_var += tf.square(delta) * self._count * self._batch_count / new_count
        new_var /= new_count
        self._update_count = self._count.assign(new_count)
        self._update_mean = self._mean.assign(new_mean)
        self._update_var = self._var.assign(new_var)

        return (X - self._mean) / (tf.sqrt(self._var) + 1e-8)

    def update(self, mean, var, count):
        sess = tf.get_default_session()
        sess.run([self._update_mean, self._update_var], feed_dict={self._batch_count:count, self._batch_mean:mean, self._batch_var:var})
        sess.run(self._update_count, feed_dict={self._batch_count:count})
