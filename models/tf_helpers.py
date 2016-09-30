from tensorflow.python.ops import variable_scope
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

_linear = rnn_cell._linear


def weight_and_bias(in_size, out_size, scope, include_bias=True):
    with variable_scope.variable_scope(scope):
        b = np.sqrt(6.0) / np.sqrt(in_size + out_size)
        weight = tf.random_uniform([in_size, out_size], minval=-1*b, maxval=b)
        if include_bias:
            bias = tf.constant(0.0, dtype=tf.float32, shape=[out_size])
            return tf.Variable(weight, name="W"), tf.Variable(bias, name="bias")
        else:
            return tf.Variable(weight, name="W")


class MemoryCellWrapper(rnn_cell.RNNCell):
    """Basic attention cell wrapper.

    Implementation based on https://arxiv.org/pdf/1601.06733.pdf.
    """

    def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
               input_size=None, state_is_tuple=False):
        """Create a cell with attention.

        Args:
            cell: an RNNCell, an attention is added to it.
            attn_length: integer, the size of an attention window.
            attn_size: integer, the size of an attention vector. Equal to
                cell.output_size by default.
            attn_vec_size: integer, the number of convolutional features calculated
                on attention state and a size of the hidden layer built from
                base cell state. Equal attn_size to by default.
            input_size: integer, the size of a hidden linear layer,
                built from inputs and attention. Derived from the input tensor
                by default.
            state_is_tuple: If True, accepted and returned states are n-tuples, where
                `n = len(cells)`.  By default (False), the states are all
                concatenated along the column axis.

        Raises:
            TypeError: if cell is not an RNNCell.
            ValueError: if cell returns a state tuple but the flag
                `state_is_tuple` is `False` or if attn_length is zero or less.
        """
        if not isinstance(cell, rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        if nest.is_sequence(cell.state_size) and not state_is_tuple:
            raise ValueError("Cell returns tuple of states, but the flag "
                             "state_is_tuple is not set. State size is: %s"
                             % str(cell.state_size))
        if attn_length <= 0:
            raise ValueError("attn_length should be greater than zero, got %s"
                             % str(attn_length))
        if not state_is_tuple:
            tf.logging.warn(
                "%s: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True." % self)
        if attn_size is None:
            attn_size = cell.output_size
        if attn_vec_size is None:
            attn_vec_size = attn_size
        self._state_is_tuple = state_is_tuple
        self._cell = cell
        self._attn_vec_size = attn_vec_size
        self._input_size = input_size
        self._attn_size = attn_size
        self._attn_length = attn_length

    @property
    def state_size(self):
        size = (self._cell.state_size, self._attn_size,
                self._attn_size * self._attn_length)
        if self._state_is_tuple:
            return size
        else:
            return sum(list(size))

    @property
    def output_size(self):
        return self._attn_size

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell with attention (LSTMA)."""
        with vs.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                state, attns, attn_states = state
            else:
                states = state
                state = array_ops.slice(states, [0, 0], [-1, self._cell.state_size])
                attns = array_ops.slice(
                    states, [0, self._cell.state_size], [-1, self._attn_size])
                attn_states = array_ops.slice(
                    states, [0, self._cell.state_size + self._attn_size],
                    [-1, self._attn_size * self._attn_length])
            attn_states = array_ops.reshape(attn_states, [-1, self._attn_length, self._attn_size])
            input_size = self._input_size
            if input_size is None:
                input_size = inputs.get_shape().as_list()[1]
            inputs = _linear([inputs, attns], input_size, True)
            lstm_output, new_state = self._cell(inputs, state)
            if self._state_is_tuple:
                new_state_cat = array_ops.concat(1, nest.flatten(new_state))
            else:
                new_state_cat = new_state
            new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
            with vs.variable_scope("AttnOutputProjection"):
                output = _linear([lstm_output, new_attns], self._attn_size, True)
            new_attn_states = array_ops.concat(1, [new_attn_states,
                                                 array_ops.expand_dims(output, 1)])
            new_attn_states = array_ops.reshape(
              new_attn_states, [-1, self._attn_length * self._attn_size])
            new_state = (new_state, new_attns, new_attn_states)
            if not self._state_is_tuple:
                new_state = array_ops.concat(1, list(new_state))
            return output, new_state

    def _attention(self, query, attn_states):
        with vs.variable_scope("Attention"):
            k = vs.get_variable("AttnW", [1, 1, self._attn_size, self._attn_vec_size])
            v = vs.get_variable("AttnV", [self._attn_vec_size])
            hidden = array_ops.reshape(attn_states,
                                     [-1, self._attn_length, 1, self._attn_size])
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            y = _linear(query, self._attn_vec_size, True)
            y = array_ops.reshape(y, [-1, 1, 1, self._attn_vec_size])
            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [2, 3])
            a = tf.nn.softmax(s)
            d = tf.reduce_sum(
                array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
            new_attns = array_ops.reshape(d, [-1, self._attn_size])
            new_attn_states = array_ops.slice(attn_states, [0, 1, 0], [-1, -1, -1])
            return new_attns, new_attn_states