from tensorflow.python.ops import variable_scope
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
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


class LSTMNCell(rnn_cell.RNNCell):

    def __init__(self, num_units, use_peepholes=False, cell_clip=None,
                 initializer=None, forget_bias=1.0, activation=tanh):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell
          input_size: Deprecated and unused.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
          activation: Activation function of the inner states.
        """
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._activation = activation
        self._state_size = rnn_cell.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
          scope: VariableScope for the created subgraph; defaults to "LSTMCell".

        Returns:
          A tuple containing:
          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        (c_prev, m_prev) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with vs.variable_scope(scope or type(self).__name__,
                               initializer=self._initializer):  # "LSTMCell"
            concat_w = rnn_cell._get_concat_variable(
                "W", [input_size.value + num_proj, 4 * self._num_units], dtype, self._num_unit_shards)

            b = vs.get_variable(
                  "B", shape=[4 * self._num_units], initializer=array_ops.zeros_initializer, dtype=dtype)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            cell_inputs = array_ops.concat(1, [inputs, m_prev])
            lstm_matrix = tf.nn_ops.bias_add(tf.math_ops.matmul(cell_inputs, concat_w), b)
            i, j, f, o = array_ops.split(1, 4, lstm_matrix)

            # Diagonal connections
            if self._use_peepholes:
                w_f_diag = vs.get_variable(
                    "W_F_diag", shape=[self._num_units], dtype=dtype)
                w_i_diag = vs.get_variable(
                    "W_I_diag", shape=[self._num_units], dtype=dtype)
                w_o_diag = vs.get_variable(
                    "W_O_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
                     sigmoid(i + w_i_diag * c_prev) * self._activation(j))
            else:
                c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                     self._activation(j))

            if self._cell_clip is not None:
                c = tf.clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            if self._use_peepholes:
                m = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                m = sigmoid(o) * self._activation(c)

            if self._num_proj is not None:
                concat_w_proj = rnn_cell._get_concat_variable(
                    "W_P", [self._num_units, self._num_proj], dtype, self._num_proj_shards)

                m = tf.math_ops.matmul(m, concat_w_proj)
                if self._proj_clip is not None:
                    m = tf.clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)

            new_state = rnn_cell.LSTMStateTuple(c, m)

            return m, new_state


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