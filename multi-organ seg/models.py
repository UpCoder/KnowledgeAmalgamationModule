import tensorflow as tf
#from tensorflow import keras
from data import *
from deeplab import *
from keras.callbacks import *
from keras.layers import *
from keras.models import Model
from keras.losses import *
from keras.backend import *
from keras import optimizers
from dataset import return_part_mnist
import numpy as np
from modules import FeatureAlignmentModule, tf_put_text
from callbacks import Tensorboard
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

def build_base_model(name='liver', restore_path=None, trainable=True,input_shape=(512,512,2),classes=2,
                     input_tensor=None):
    with tf.variable_scope(name):
        block_outputs = [] #每一层中间的输出
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            img_input = input_tensor
        model = Deeplabv3(input_tensor=img_input,classes=classes)
        layers = model.layers
        for layer in layers:
            layer.name = layer.name + name
        block_outputs.append(
            model.get_layer('Channel1expanded_conv_2_project_BN' + name).output
        )
        block_outputs.append(
            model.get_layer('Channel1expanded_conv_5_project_BN' + name).output
        )
        block_outputs.append(
            model.get_layer('Channel1expanded_conv_9_project_BN' + name).output
        )
        block_outputs.append(
            model.get_layer('Channel1expanded_conv_12_project_BN' + name).output
        )
        block_outputs.append(
            model.get_layer('Channel1expanded_conv_15_project_BN' + name).output
        )
        #logits = classifier_layer(tf.keras.layers.GlobalAveragePooling2D()(block_outputs[-1])) #最后一层……看看怎么处理
        logits1 = model.get_layer('custom_logits_semantic' + name).output
        logits2 = Conv2D(classes, 1,1,kernel_regularizer= regularizers.l2(1e-5), activation='sigmoid')(logits1)
        logits = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(logits2)
        if restore_path is not None:
            model.load_weights(restore_path)
        model.trainable = trainable
        return model, block_outputs, logits

def cal_soft_loss(logits_t, logits_s):
    soft_loss = mean_squared_error(logits_t, logits_s)
    return soft_loss

def cal_entropy(logits):
    print('logits is ',logits)
    #probs = Conv2D(1, 1, kernel_regularizer=regularizers.l2(1e-5), activation='sigmoid')(logits)
    #print(probs)
    #entropy_layer = Lambda(
    #    lambda x: sum(-x * tf.log(tf.clip_by_value(x, 1e-8, 1.0)), axis=3))
    entropy_layer = Lambda(lambda x: tf.sigmoid(x[:, :, :, 0]/x[:, :, :, 1]))
    entropy = entropy_layer(logits)
    print('entropy is ',entropy)
    return entropy

def train_student_net():
    path = './multi_train'
    data_gen_args = dict(rotation_range=0.4,
                         width_shift_range=0,
                         height_shift_range=0,
                         shear_range=0,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    trainGene = trainGenerator_unsup(7, path, 'img', data_gen_args, save_to_dir = None)

    input_shape = (512, 512, 2)
    input_tensor = Input(shape=input_shape)
    with tf.device('/gpu:1'):
        teacher_net1, t1_output_blocks, t1_logits = build_base_model(name='liver',restore_path='./liver.h5', trainable=False,input_tensor=input_tensor)
    with tf.device('/gpu:2'):
        teacher_net2, t2_output_blocks, t2_logits = build_base_model(name='kidney',restore_path='./kidney.h5', trainable=False,input_tensor=input_tensor)
    with tf.device('/gpu:3'):
        student_net, s_output_blocks, s_logits = build_base_model(name='multi', classes=4, input_tensor=input_tensor)

    with tf.device('/cpu:0'):
        t1_entropy = cal_entropy(t1_logits)
        t2_entropy = cal_entropy(t2_logits)

        t1_blocks = [FeatureAlignmentModule(block.get_shape().as_list()[-1]) for block in t1_output_blocks]
        t2_blocks = [FeatureAlignmentModule(block.get_shape().as_list()[-1]) for block in t2_output_blocks]
        t1_cb_outs = [t1_blocks[i].call(t1_output_blocks[i], s_output_blocks[i]) for i in range(len(t1_output_blocks))]
        t2_cb_outs = [t2_blocks[i].call(t2_output_blocks[i], s_output_blocks[i]) for i in range(len(t2_output_blocks))]

        t1_cb_losses = [FeatureAlignmentModule.block_loss(*t1_cb_outs[i]) for i in range(len(t1_cb_outs))]
        t2_cb_losses = [FeatureAlignmentModule.block_loss(*t2_cb_outs[i]) for i in range(len(t2_cb_outs))]
        t1_cb_losses = tf.add_n(t1_cb_losses) / len(t1_cb_losses)
        t2_cb_losses = tf.add_n(t2_cb_losses) / len(t2_cb_losses) #正则化

        print(t1_entropy,t2_entropy,t1_cb_losses,t2_cb_losses)
        block_loss_layer = Lambda(
            lambda x: tf.where(tf.keras.backend.less_equal(x[0], x[1]),
                                                             x[2], x[3], name='cal_block_loss'))

        block_loss_tensor = block_loss_layer([t1_entropy, t2_entropy, t1_cb_losses, t2_cb_losses])

        # block_loss_tensor = tf.where(tf.keras.backend.less_equal(t1_entropy, t2_entropy),t1_cb_losses, t2_cb_losses)
        # 开始计算soft loss
        # print('t1_logits is ', t1_logits)
        # [5,2]
        new_t1_logits = concatenate([t1_logits, zeros_like(t1_logits)], axis=3)
        new_t2_logits = concatenate([zeros_like(t2_logits), t2_logits], axis=3)

        print(t1_entropy, new_t1_logits, s_logits)
        soft_loss = tf.where(tf.keras.backend.less_equal(t1_entropy, t2_entropy), cal_soft_loss(new_t1_logits, s_logits),
                                cal_soft_loss(new_t2_logits, s_logits))
        soft_loss = soft_loss * 200.
        print('soft loss is ', soft_loss)
        tf.summary.scalar('soft_loss', tf.reduce_mean(soft_loss))

        total_loss_layer = Lambda(
            lambda x: tf.reduce_mean(x[0] + x[1]))
        total_loss_tensor = total_loss_layer([block_loss_tensor, soft_loss])

        img_inputs = get_source_inputs(input_tensor)
        target_model = Model(inputs=img_inputs, outputs=[t1_logits, t2_logits, s_logits])
        target_model.add_loss(total_loss_tensor)
        target_model.compile(optimizer=optimizers.Adam(lr=0.0001))
        #tensorboard_callback = Tensorboard(summary_op=tf.summary.merge_all(), log_dir='./log/', batch_interval=5)
        #target_model.fit_generator(None, None, epochs=5, verbose=1, batch_size=batch_size, steps_per_epoch=24704/batch_size,callbacks=[tensorboard_callback])
        csv_logger = CSVLogger('./tmp/multi_seg1.csv', append=True)
        model_checkpoint = ModelCheckpoint('./Models/multi_seg1-{epoch:02d}.h5',
                                           verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
        target_model.fit_generator(generator=trainGene, steps_per_epoch=3000, epochs=100, callbacks=[model_checkpoint, csv_logger])  # , max_queue_size=10, workers=3, use_multiprocessing=True)
        target_model.save_weights('./multi.h5')

def evaluate_component():

    test_path = './multi_train/'
    org_input_tensor = Input(shape=(512, 512, 2))
    student_net, s_output_blocks, s_logits = build_base_model(name='multi',trainable=False, classes=4, input_tensor=org_input_tensor)

    student_net.load_weights('./multi.h5', by_name=True)
    testGene = testGenerator(test_path)
    results = student_net.predict_generator(testGene, 1746)
    save_path = os.path.join(test_path, 'multi1')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saveResult(save_path, results, flag_multi_class = True, num_class=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=int, default=1, help='the step ID, 1 or 2')
    args = vars(parser.parse_args())
    if args['step'] == 1:
        train_student_net()
    elif args['step'] == 2:
        evaluate_component()
    else:
        assert False

