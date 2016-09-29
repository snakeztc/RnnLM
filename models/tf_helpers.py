from tensorflow.python.ops import variable_scope
import numpy as np
import tensorflow as tf


def weight_and_bias(in_size, out_size, scope, include_bias=True):
    with variable_scope.variable_scope(scope):
        b = np.sqrt(6.0) / np.sqrt(in_size + out_size)
        weight = tf.random_uniform([in_size, out_size], minval=-1*b, maxval=b)
        if include_bias:
            bias = tf.constant(0.0, dtype=tf.float32, shape=[out_size])
            return tf.Variable(weight, name="W"), tf.Variable(bias, name="bias")
        else:
            return tf.Variable(weight, name="W")