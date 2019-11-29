import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


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


class CustomCheckpointer(tf.keras.callbacks.Callback):
    def __init__(self, filepath, custom_model, monitor, mode, save_best_only,
                 verbose=0, verbose_description='encoder', batch_interval=10, prefix=''):
        self.filepath = filepath
        self.custom_model = custom_model
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.description = verbose_description
        self.batch_interval = batch_interval
        self.prefix = prefix

        print('Initializing custom checkpointer for model `{}`.'.format(
            self.custom_model.name))
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.best = np.Inf if mode == 'min' else -np.Inf
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if not self.save_best_only or self.monitor_op(current, self.best):
            if self.verbose > 0:
                print('Saving the custom {} model to {}'.format(
                    self.description, self.filepath))
            self.best = current
            cur_file_path = self.filepath + '{}-ep{:02d}-End-loss{:.4f}.h5'.format(self.prefix, self.epoch,
                                                                                   np.mean(current))
            self.custom_model.save_weights(cur_file_path, overwrite=True)

    # def on_batch_end(self, batch, logs=None):
    #     current = logs.get(self.monitor)
    #     # print current
    #     if batch % self.batch_interval == 0:
    #         if not self.save_best_only:
    #             if self.verbose > 0:
    #                 print('Saving the custom {} model to {}'.format(
    #                     self.description, self.filepath))
    #             self.best = current
    #             cur_file_path = self.filepath + 'ep{:02d}-batch{:06d}-loss{:.4f}.h5'.format(self.epoch, batch,
    #                                                                                         np.mean(current))
    #             # print self.custom_model, cur_file_path
    #             self.custom_model.save_weights(cur_file_path, overwrite=True)
    #         else:
    #             if self.monitor_op(current, self.best):
    #                 if self.verbose > 0:
    #                     print('Saving the custom {} model to {}'.format(
    #                         self.description, self.filepath))
    #                 self.best = current
    #                 cur_file_path = self.filepath + 'ep{:02d}-batch{:06d}-loss{:.4f}.h5'.format(self.epoch, batch,
    #                                                                                             np.mean(current))
    #                 self.custom_model.save_weights(cur_file_path, overwrite=True)