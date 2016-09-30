import os
import time

import numpy as np
import tensorflow as tf
from models.StateDataLoader import StateDataLoader
from models.StateLM import StateLM
from PTB.PTBCorpus import PTBCorpus

# constants
tf.app.flags.DEFINE_string("data_dir", "PTB/ptb-lm", "Raw data directory.")
tf.app.flags.DEFINE_string("work_dir", "seq_working", "Experiment results directory.")
tf.app.flags.DEFINE_integer("embedding_size", 150, "The embedding size of word embedding")
tf.app.flags.DEFINE_integer("cell_size", 400, "The width of RNN")
tf.app.flags.DEFINE_integer("batch_size", 16, "Number of sample each mini batch")
tf.app.flags.DEFINE_integer("max_sent_len", 25, "Max number of turn to be modelled")
tf.app.flags.DEFINE_integer("num_layers", 2, "The number of layers in recurrent neural network")
tf.app.flags.DEFINE_integer("max_epoch", 200, "Max number of turn to be modelled")
tf.app.flags.DEFINE_float("l2_coef", 1e-5, "L2 regulzation weight for weight matrixes")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate of SGD")
tf.app.flags.DEFINE_float("momentum", 0.9, "How much momentum")
tf.app.flags.DEFINE_bool("use_dropout", True, "if use drop out")
tf.app.flags.DEFINE_float("improve_threshold", 0.995, "how much decrease in dev loss counts")
tf.app.flags.DEFINE_float("patience_increase", 2.0, "How much more we wait for a new discovered minmum")
tf.app.flags.DEFINE_bool("early_stop", True, "Whether to early stop")
FLAGS = tf.app.flags.FLAGS


# get data set
api = PTBCorpus(FLAGS.data_dir)
corpus = api.get_corpus()

train_data, valid_data, test_data = corpus.get("train"), corpus.get("valid"), corpus.get("test")

# convert to numeric input outputs that fits into TF models
train_feed = StateDataLoader("train", train_data)
valid_feed = StateDataLoader("valid", valid_data)
test_feed = StateDataLoader("test", test_data)

log_dir = os.path.join(FLAGS.work_dir, "run"+str(int(time.time())))

# begin training
with tf.Session() as sess:
    model = StateLM(sess,
                    vocab_size=api.get_vocab_size(),
                    cell_size=FLAGS.cell_size,
                    embedding_size=FLAGS.embedding_size,
                    num_layer=FLAGS.num_layers,
                    log_dir=log_dir,
                    learning_rate=FLAGS.learning_rate,
                    momentum=FLAGS.momentum,
                    use_dropout=FLAGS.use_dropout,
                    l2_coef=FLAGS.l2_coef)

    ckp_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(ckp_dir):
        os.mkdir(ckp_dir)
    ckpt = tf.train.get_checkpoint_state(ckp_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading models parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created models with fresh parameters.")
        sess.run(tf.initialize_all_variables())

    global_t = 0
    patience = 10 # wait for at least 10 epoch before stop
    best_dev_loss = np.inf
    checkpoint_path = os.path.join(ckp_dir, "state-ptb-lm.ckpt")

    for epoch in range(FLAGS.max_epoch):
        train_feed.epoch_init(FLAGS.batch_size)
        print("epoch %d" % epoch)
        global_t, losses = model.train(global_t, sess, train_feed)
        train_loss = np.mean(losses)
        print("Train loss for %f and perplexity %f" % (train_loss, np.exp(train_loss)))

        # begin validation
        valid_feed.epoch_init(FLAGS.batch_size, shuffle=False)
        losses = model.valid(global_t, sess, valid_feed)
        valid_loss = np.mean(losses)
        print("Valid loss for %f and perplexity %f" % (valid_loss, np.exp(valid_loss)))

        # only save a models if the dev loss is smaller
        done_epoch = epoch +1
        if valid_loss < best_dev_loss:
            if valid_loss <= best_dev_loss * FLAGS.improve_threshold:
                patience = max(patience, done_epoch *FLAGS.patience_increase)

            # still save the best train model
            model.saver.save(sess, checkpoint_path, global_step=epoch)
            best_dev_loss = valid_loss
        if FLAGS.early_stop and patience <= done_epoch:
            print("!!Early stop due to run out of patience!!")
            break
    print("Best dev loss %f" % best_dev_loss)
    print("Done training")












