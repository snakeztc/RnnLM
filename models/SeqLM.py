import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import tf_helpers
import os
import time
import utils

from tensorflow.python.ops.nn import rnn_cell


class SeqLM(object):
    """
    This models treat LM as a sequential labelling problem, where it needs to make prediction at every step
    """
    def __init__(self, sess, vocab_size, cell_size, embedding_size, num_layer, log_dir,
                 learning_rate=0.001, momentum=0.9, use_dropout=True, l2_coef=1e-6):

        with tf.name_scope("io"):
            self.inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name="prev_words")
            self.input_lens = tf.placeholder(dtype=tf.int32, shape=(None, ), name="sent_len")
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None, None), name="next_words")
            self.keep_prob =tf.placeholder(dtype=tf.float32, name="keep_prob")

        max_sent_len = array_ops.shape(self.labels)[1]
        with variable_scope.variable_scope("word-embedding"):
            embedding = tf_helpers.weight_and_bias(vocab_size, embedding_size, "embedding_w", include_bias=False)
            input_embedding = embedding_ops.embedding_lookup(embedding, tf.squeeze(tf.reshape(self.inputs, [-1, 1]),
                                                                                   squeeze_dims=[1]))

            input_embedding = tf.reshape(input_embedding, [-1, max_sent_len, embedding_size])

        with variable_scope.variable_scope("rnn"):
            cell = tf_helpers.MemoryCellWrapper(
                rnn_cell.LSTMCell(cell_size, use_peepholes=True, state_is_tuple=True),
                attn_length=5, state_is_tuple=True)

            if use_dropout:
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            if num_layer > 1:
                cell = rnn_cell.MultiRNNCell([cell] * num_layer, state_is_tuple=True)

            # add output projection
            cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, vocab_size)

            # and enc_last_state will be same as the true last state
            self.logits, last_state = tf.nn.dynamic_rnn(
                cell,
                input_embedding,
                dtype=tf.float32,
                sequence_length=self.input_lens,
            )
        vars = tf.trainable_variables()
        self.loss = self.sequence_loss()
        tf.scalar_summary("entropy_loss", self.loss)
        tf.scalar_summary("perplexity" ,tf.exp(self.loss))
        self.summary_op = tf.merge_all_summaries()

        # weight decay
        loss_l2= tf.add_n([tf.nn.l2_loss(v) for v in vars if "bias" not in v.name.lower()])
        self.reg_loss = self.loss + l2_coef * loss_l2

        # optimization
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_ops = optimizer.minimize(self.reg_loss)

        train_log_dir = os.path.join(log_dir, "train")
        valid_log_dir = os.path.join(log_dir, "valid")
        print "Save summary to %s" % log_dir
        self.train_summary_writer = tf.train.SummaryWriter(train_log_dir, sess.graph)
        self.valid_summary_writer = tf.train.SummaryWriter(valid_log_dir, sess.graph)
        self.saver = tf.train.Saver(tf.all_variables())

    def sequence_loss(self):
        with ops.name_scope("sequence_loss_by_example"):
            weights = tf.to_float(tf.sign(self.labels, name="mask"))
            batch_size = array_ops.shape(self.labels)[0]
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(self.logits, self.labels)
            log_perps = tf.reduce_sum(tf.reduce_sum(crossent * weights, reduction_indices=1) / tf.to_float(self.input_lens))
        return log_perps / tf.to_float(batch_size)

    def train(self, global_t, sess, train_feed):
        losses = []
        local_t = 0
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            inputs, input_lens, outputs = batch
            feed_dict = {self.inputs: inputs, self.input_lens: input_lens, self.labels: outputs, self.keep_prob: 0.5}
            _, loss, summary = sess.run([self.train_ops, self.loss, self.summary_op], feed_dict)
            self.train_summary_writer.add_summary(summary, global_t)
            losses.append(loss)
            global_t += 1
            local_t += 1
            if local_t % 50 == 0:
                utils.progress(local_t/float(train_feed.num_batch))
        return global_t, losses

    def valid(self, t, sess, valid_feed):
        losses = []
        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            inputs, input_lens, outputs = batch
            feed_dict = {self.inputs: inputs, self.input_lens: input_lens, self.labels: outputs, self.keep_prob: 1.0}
            loss, summary = sess.run([self.loss, self.summary_op], feed_dict)
            self.valid_summary_writer.add_summary(summary, t)
            losses.append(loss)

        return losses



