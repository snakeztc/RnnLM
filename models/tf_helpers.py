from tensorflow.python.ops import variable_scope
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops

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


class MemoryLSTMCell(rnn_cell.RNNCell):

    def __init__(self, num_units, attn_length, use_peepholes=False, cell_clip=None,
                 initializer=None, forget_bias=1.0, activation=tanh):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell
          attn_length: the size of attention window for non-markov update
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
        self._attn_length = attn_length
        self._output_size = num_units

    @property
    def state_size(self):
        # h_summary, c_tape, h_tape
        return self._attn_length, self._num_units, self._num_units*self._attn_length, self._num_units*self._attn_length

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run one step of MemoryLSTM.
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
        (a, h_prev_summary, c_tape_prev, h_tape_prev) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        with vs.variable_scope(scope or type(self).__name__, initializer=self._initializer):  # "LSTMCell"
            concat_w = rnn_cell._get_concat_variable(
                "W", [input_size.value + self._num_units, 4 * self._num_units], dtype, 1)

            b = vs.get_variable("Bias", shape=[4 * self._num_units], initializer=array_ops.zeros_initializer, dtype=dtype)

            # reshape tape to 3D
            c_tape_prev = array_ops.reshape(c_tape_prev, [-1, self._attn_length, self._num_units])
            h_tape_prev = array_ops.reshape(h_tape_prev, [-1, self._attn_length, self._num_units])

            a, new_c_summary, new_h_summary = self._attention(inputs, h_prev_summary, c_tape_prev, h_tape_prev)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            cell_inputs = array_ops.concat(1, [inputs, new_h_summary])
            lstm_matrix = tf.nn.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
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
                c = (sigmoid(f + self._forget_bias + w_f_diag * new_c_summary) * new_c_summary +
                     sigmoid(i + w_i_diag * new_c_summary) * self._activation(j))
            else:
                c = (sigmoid(f + self._forget_bias) * new_c_summary + sigmoid(i) *
                     self._activation(j))

            if self._cell_clip is not None:
                c = tf.clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

            if self._use_peepholes:
                h = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                h = sigmoid(o) * self._activation(c)

            # remove old value
            new_h_tape = array_ops.slice(h_tape_prev, [0, 1, 0], [-1, -1, -1])
            new_c_tape = array_ops.slice(c_tape_prev, [0, 1, 0], [-1, -1, -1])

            # append the new c and h to the tape
            new_c_tape = array_ops.concat(1, [new_c_tape, array_ops.expand_dims(c, 1)])
            new_h_tape = array_ops.concat(1, [new_h_tape, array_ops.expand_dims(h, 1)])

            # flatten the tape to 2D
            new_c_tape = array_ops.reshape(new_c_tape, [-1, self._attn_length * self._num_units])
            new_h_tape = array_ops.reshape(new_h_tape, [-1, self._attn_length * self._num_units])

            new_state = (a, new_h_summary, new_c_tape, new_h_tape)

            return h, new_state

    def _attention(self, x, h_prev_summary, c_tape, h_tape):
        """
        :param x: batch_size * input_size
        :param h_prev_summary:batch_size * cell_size
        :param c_tape: batch_size * memory_size * cell_size
        :param h_tape: batch_size * memory_size * cell_size
        :return:
        """
        input_size = x.get_shape().with_rank(2)[1]

        with vs.variable_scope("Attention"):
            # mask out empty slots
            mask = tf.sign(tf.reduce_max(tf.abs(h_tape), reduction_indices=2))

            # construct query for attention
            concat_w = rnn_cell._get_concat_variable("query_w", [input_size.value+self._num_units, self._num_units], x.dtype, 1)
            b = vs.get_variable("query_bias", shape=[self._num_units], initializer=array_ops.zeros_initializer, dtype=x.dtype)
            query = tf.nn.bias_add(math_ops.matmul(array_ops.concat(1, [x, h_prev_summary]), concat_w), b)
            query = array_ops.reshape(query, [-1, 1, 1, self._num_units])

            # get the weights for attention
            k = vs.get_variable("AttnW", [1, 1, self._num_units, self._num_units])
            v = vs.get_variable("AttnV", [self._num_units])
            hidden = array_ops.reshape(h_tape, [-1, self._attn_length, 1, self._num_units])
            memory = array_ops.reshape(c_tape, [-1, self._attn_length, 1, self._num_units])

            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            s = tf.reduce_sum(v * tf.tanh(hidden_features + query), [2, 3])
            a = tf.nn.softmax(s) * mask
            a = a / (tf.reduce_sum(a, reduction_indices=1, keep_dims=True) + 1e-12)

            h_summary = tf.reduce_sum(array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
            c_summary = tf.reduce_sum(array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * memory, [1, 2])

            return a, c_summary, h_summary


class MemoryGRUCell(rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, attn_length, attn_size=None, input_size=None, activation=tanh):
        if input_size is not None:
            tf.logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._attn_length = attn_length
        self._activation = activation
        if attn_size is not None:
            self._attn_size = attn_size
        else:
            self._attn_size = num_units
        self._attn_indexes = tf.reshape(tf.to_float(tf.range(1, self._attn_length + 1)) / self._attn_length,
                                        [1, self._attn_length, 1])

    @property
    def state_size(self):
        return self._num_units, self._attn_length*self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with n_units cells."""
        (h_prev_summary, h_tape_prev) = state
        with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
            h_tape_prev = array_ops.reshape(h_tape_prev, [-1, self._attn_length, self._num_units])

            # get new summary
            h_summary = self._attention(inputs, h_prev_summary, h_tape_prev)

            with vs.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = array_ops.split(1, 2, _linear([inputs, h_summary],
                                                 2 * self._num_units, True, 1.0))
                r, u = sigmoid(r), sigmoid(u)
            with vs.variable_scope("Candidate"):
                c = self._activation(_linear([inputs, r * h_summary], self._num_units, True))

            new_h = u * h_summary + (1 - u) * c

            # append the new h to the tape
            new_h_tape = array_ops.slice(h_tape_prev, [0, 1, 0], [-1, -1, -1])
            new_h_tape = array_ops.concat(1, [new_h_tape, array_ops.expand_dims(new_h, 1)])
            new_h_tape = array_ops.reshape(new_h_tape, [-1, self._attn_length * self._num_units])

        return new_h, (h_summary, new_h_tape)

    def _attention(self, x, h_prev_summary, h_tape):
        """
        :param x: batch_size * input_size
        :param h_prev_summary:batch_size * cell_size
        :param h_tape: batch_size * memory_size * cell_size
        :return: weighted sum h, and trucated h_tape
        """
        input_size = x.get_shape().with_rank(2)[1]

        with vs.variable_scope("Attention"):
            # construct query for attention
            concat_w = rnn_cell._get_concat_variable("query_w", [input_size.value+self._num_units, self._attn_size], x.dtype, 1)
            b = vs.get_variable("query_bias", shape=[self._attn_size], initializer=array_ops.zeros_initializer, dtype=x.dtype)
            query = tf.nn.bias_add(math_ops.matmul(array_ops.concat(1, [x, h_prev_summary]), concat_w), b)
            query = array_ops.reshape(query, [-1, 1, 1, self._attn_size])

            # get temporal feature
            mask = tf.sign(tf.reduce_max(tf.abs(h_tape), reduction_indices=2, keep_dims=True))
            t_ids = tf.ones_like(mask) * self._attn_indexes
            mask = tf.squeeze(mask)

            # get the weights for attention
            k = vs.get_variable("AttnW", [1, 1, self._num_units+1, self._attn_size])
            v = vs.get_variable("AttnV", [self._attn_size])
            hidden = tf.concat(2, [h_tape, t_ids])
            hidden = array_ops.reshape(hidden, [-1, self._attn_length, 1, self._num_units+1])

            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")

            # compute attention point
            s = tf.reduce_sum(v * tf.tanh(hidden_features + query), [2, 3])
            a = tf.nn.softmax(s)* mask
            a = a / (tf.reduce_sum(a, reduction_indices=1, keep_dims=True) + 1e-12)

            h_summary = tf.reduce_sum(array_ops.reshape(a, [-1, self._attn_length, 1]) * h_tape, reduction_indices=1)

            return h_summary
