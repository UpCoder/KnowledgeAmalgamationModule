import tensorflow as tf
from tensorflow.keras import backend as K


class Tensorboard(tf.keras.callbacks.Callback):
    def __init__(self,
                 summary_op,
                 batch_interval=10,
                 log_dir='./log',
                 batch_size=64,
                 write_graph=True):
        self.write_graph = write_graph
        self.batch_interval = batch_interval
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir)
        self.sess = K.get_session()
        self.summary_op = summary_op
        self.epoch = 0
        self.steps_per_epoch = 24704 // batch_size

    def on_batch_begin(self, batch, logs=None):
        self.step = self.epoch * self.steps_per_epoch + batch
        logs.update({'learning_rate': float(K.get_value(self.model.optimizer.lr))})
        if batch % self.batch_interval == 0:
            summary_value = self.sess.run(self.summary_op)
            self.writer.add_summary(summary_value, self.step)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch