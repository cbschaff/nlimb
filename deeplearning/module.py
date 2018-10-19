import tensorflow as tf
from deeplearning import tf_util as U
import numpy as np
import os, pickle

class Module(object):
    """
    Base class for Neural Networks blocks that manages temporal states
    and nicely interfaces with the TF graph.
    """
    # Set this in subclasses to control the number of inputs to the module
    ninputs = None
    def __init__(self, name, *modules):
        """
        Inputs:
         - name: the name of the module. Sets the variable scope.
         - modules: a list of Modules whose outputs will be the inputs to this module. Use the Input Module to define placeholders and other data feeds.
        """
        if self.ninputs is not None:
            assert len(modules) == self.ninputs, "Incorrect number of Inputs. Expected: {}, Received: {}".format(self.ninputs, len(modules))

        for module in modules:
            assert isinstance(module, Module)
        self.modules = modules
        self.state_modules = []
        self._built = False
        self.name = name
        self.is_recurrent = np.array([m.is_recurrent for m in self.modules]).any()

    def build(self, scope, nbatch, nstep=1, reuse=False, trainable=None):
        """
        Builds the TF graph for this module.
        """
        if self._built:
            return
        self._built = True
        self.scope = scope
        self.nbatch = nbatch
        self.nstep = nstep

        self.placeholders = []
        self.state_placeholders = []
        self.state_out = []
        self.h = [] # tf tensor for the direct inputs of this module
        self.state = [] # tf tensor for the input state of this module
        for module in self.modules:
            module.build(scope, nbatch, nstep, reuse, trainable)
            for x in module.placeholders:
                if x not in self.placeholders:
                    self.placeholders.append(x)
            for x in module.state_placeholders:
                if x not in self.state_placeholders:
                    self.state_placeholders.append(x)
            for x in module.state_out:
                if x not in self.state_out:
                    self.state_out.append(x)
            self.h.extend(module.out)

        for module in self.state_modules:
            module.build(scope, nbatch, nstep, reuse, trainable)
            assert len(module.state_placeholders) == 0, "state_modules are not allowed to have their own temporal state."
            assert len(module.state_out) == 0, "state_modules are not allowed to have their own temporal state."
            assert len(module.state) == 0, "state_modules are not allowed to have their own temporal state."
            for x in module.placeholders:
                if x not in self.state_placeholders:
                    self.state_placeholders.append(x)
            self.state.extend(module.out)

        if trainable is not None:
            def custom_getter(getter, name, *args, **kwargs):
                new_kwargs = {k:v for k,v in kwargs.items() if k != 'trainable'}
                return getter(name, *args, trainable=trainable, **new_kwargs)
        else:
            custom_getter = None

        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope(self.name, reuse=reuse, custom_getter=custom_getter):
                self.make_graph(self.h, self.state)

    def make_graph(self, inputs, state):
        # hide temoral state for non-recurrent modules
        out = self._build(inputs)
        self.out = self._check_list(out)

    def _build(self, inputs):
        """
        OVERWRITE this function in your subclasses.
        Builds the TF graph for this module.
        Inputs:
          - inputs: a list of input tensors
        Outputs:
          - The outputs of the module: A list of tensors.
          - The new temporal state: A list of tensors.

        See layers.py for examples.
        """
        raise NotImplementedError

    def _check_list(self, inputs):
        if isinstance(inputs, tuple):
            return list(inputs)
        if not isinstance(inputs, list):
            return [inputs]
        return inputs

    def _check_run_inputs(self, inputs):
        if isinstance(inputs, dict):
            return inputs
        return self._check_list(inputs)

    def run(self, inputs=[], state=[], out=True, state_out=True, **flags):
        """
        Runs this module with some inputs.
        Inputs:
          - inputs: A list of numpy arrays corresponding to self.placeholders. You can also pass a standard feed dictionary where self.placeholders are the keys.
          - state: Same as inputs, but for self.state_placeholders.
          - out: A flag to determine whether to return the output (self.out) of the module
          - state_out: Same as out, but for the transformed state of the model (self.state_out)
          - flags: Additional outputs of this model to return. These flags will be passed to self._add_run_args which can be overwritten to add additional tensors to the output dictionary and placeholders to the feed_dict.

        Outputs:
          - A dictionary of all of the requested outputs. Example:
                {
                  'out': [array([1])],
                  'state_out': [array([0,1,2]), array([3,4,5])]
                }
        """
        assert self._built, "You must build your network before running it."
        sess = tf.get_default_session()
        assert sess is not None, "The default session is None. Did you forget to create one?"

        def create_feed_dict(placeholders, inputs):
            if isinstance(inputs, dict):
                return inputs
            return dict(list(zip(placeholders, inputs)))

        feed_dict = create_feed_dict(self.placeholders, self._check_run_inputs(inputs))
        feed_dict.update(create_feed_dict(self.state_placeholders, self._check_run_inputs(state)))

        outs = {}
        if out: outs['out'] = self.out
        if state_out: outs['state_out'] = self.state_out
        self.add_run_args(outs, feed_dict, **flags)
        run_out = sess.run(outs, feed_dict=feed_dict)
        # For convenience, remove list for 'out' when the module only has one output
        if 'out' in run_out and isinstance(run_out['out'], list) and len(run_out['out']) == 1:
            run_out['out'] = run_out['out'][0]
        return run_out

    def add_run_args(self, outs, feed_dict, **flags):
        self._add_run_args(outs, feed_dict, **flags)
        for m in self.modules:
            m.add_run_args(outs, feed_dict, **flags)

    def _add_run_args(self, outs, feed_dict, **flags):
        """
        Overwrite this in subclasses, adding additional output args / things to the feed dict as needed.
        For each True flag, add an output to the outs dict. Every flag passed to run will get passed to every module in the network, so try to keep the keys unique (and preferably the same name as the corresponding flag)
        Example:
            def _add_run_args(self, outs, feed_dict, **flags):
                if 'logits' in flags and flags['logits']:
                    outs['logits'] = self.logits
        See Optimizer below for another example.
        """
        pass

    def __call__(self, inputs=[], state=[]):
        """
        Convenience method to get the output of a module.
        """
        return self.run(inputs, state, state_out=False)['out']

    def placeholders(self):
        """
        Returns the placeholders required to run this module.
        """
        return self.placeholders

    def state_placeholders(self):
        """
        Returns the placeholders required for the state of this module (i.e. the hidden state for an rnn).
        """
        return self.state_placeholders

    def variables(self):
        if not self._built:
            vs = []
        else:
            vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '{}_*[0-9]*/{}'.format(self.scope, self.name))
        for m in self.modules:
            vs.extend(m.variables())
        return sorted(list(set(vs)), key=lambda v: v.name)

    def trainable_variables(self):
        if not self._built:
            vs = []
        else:
            vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}_*[0-9]*/{}'.format(self.scope, self.name))
        for m in self.modules:
            vs.extend(m.trainable_variables())
        return sorted(list(set(vs)), key=lambda v: v.name)

    def find_submodules_by_instance(self, M):
        """
        Returns a list of submodules of this module
        that are instances of M.
        """
        modules = []
        for m in self.modules:
            if isinstance(m, M):
                modules.append(m)
            modules.extend(m.find_submodules_by_instance(M))
        return modules

    def find_submodules_by_name(self, name):
        """
        Returns a list of submodules of this module
        that have a given name.
        """
        modules = []
        for m in self.modules:
            if m.name == name:
                modules.append(m)
            modules.extend(m.find_submodules_by_name(name))
        return modules


class RecurrentModule(Module):
    def __init__(self, name, *modules, state_modules=[]):
        """
        Inputs:
         - See above for name and modules.
         - state_modules: Modules for the temporal state of this module. These modules cannot have their own state modules.
        """
        super().__init__(name, *modules)
        for module in state_modules:
            assert isinstance(module, Module)
            assert len(module.state_modules) == 0, "state_modules are not allowed to have their own temporal state."
        self.state_modules = state_modules
        self.is_recurrent = True

    def make_graph(self, inputs, state):
        out, state_out = self._build(inputs, state)
        self.out = self._check_list(out)
        self.state_out.extend(self._check_list(state_out))

    def _build(self, inputs, state):
        """
        OVERWRITE this function in your subclasses.
        Builds the TF graph for this module.
        Inputs:
          - input: a list of input tensors
          - state: a list of tensors for the temporal state of the module.
        Outputs:
          - The outputs of the module: A list of tensors.
          - The new temporal state: A list of tensors.

        See layers.LSTM for an example.
        """
        raise NotImplementedError


class Optimizer(Module):
    """
    Optimize the output of a module. Sets up common functions to compute gradient and update the model.
    """
    ninputs=1
    def __init__(self, name, loss):
        if not isinstance(loss, Module):
            assert False, "loss must be a Module"
        super().__init__(name, loss)
        self.loss = loss

    def make_graph(self, inputs, state):
        self.out = inputs[0] # inputs[0] == loss.out
        # build optimizer ops
        self._grad, self._update = self._build(inputs[0])

    def _build(self, loss):
        """
        OVERWRITE this function in your subclasses.
        Builds the TF graph for this module.
        Inputs:
          - a tensor for the loss function.
        Outputs:
          - a list of gradients (i.e., the output of tf.gradients).
          - an update operation.

        See layers.Adam for an example.
        """
        raise NotImplementedError

    def _add_run_args(self, outs, feed_dict, **flags):
        if 'grad' in flags and flags['grad']:
            outs['grad'] = self._grad
        if 'update' in flags and flags['update']:
            outs['update'] = self._update

    # convenience methods
    def lossandgrad(self, inputs=[], state=[]):
        outs = self.run(inputs, state, out=True, state_out=False, grad=True, update=False)
        return outs['out'], outs['grad']

    def grad(self, inputs=[], state=[]):
        return self.run(inputs, state, out=False, state_out=False, grad=True, update=False)['grad']

    def update(self, inputs=[], state=[]):
        self.run(inputs, state, out=False, state_out=False, grad=False, update=True)



#########################
# Save and load modules
#########################

def save_module(module, fname):
    if module._built:
        assert False, "Cannot save a built module."
    with open(fname, 'wb') as f:
        pickle.dump(module, f)

def load_module(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
