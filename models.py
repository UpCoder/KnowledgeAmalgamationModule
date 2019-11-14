import tensorflow as tf
from tensorflow import keras
from dataset import return_part_mnist
import numpy as np
from modules import FeatureAlignmentModule, tf_put_text
from callbacks import Tensorboard
import argparse
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)


def build_base_model(model_name='vgg16', include_top=False, classes=2, name='1_2', restore_path=None, trainable=True,
                     org_input_tensor=None):
    with tf.variable_scope(name):
        block_outputs = []
        if org_input_tensor is None:
            print('create org_input_tensor')
            org_input_tensor = tf.keras.layers.Input((28, 28, 1))
        print(org_input_tensor)
        input_tensor = tf.tile(org_input_tensor, [1, 1, 1, 3])
        resize_layer = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [128, 128]))
        input_tensor = resize_layer(input_tensor)
        if model_name == 'vgg16':
            base_model = keras.applications.vgg16.VGG16(include_top=include_top, input_tensor=input_tensor, name=name)
            layers = base_model.layers
            for layer in layers:
                layer._name = layer.name + name
            block_outputs.append(
                base_model.get_layer('block1_pool' + name).output
            )
            block_outputs.append(
                base_model.get_layer('block2_pool' + name).output
            )
            block_outputs.append(
                base_model.get_layer('block3_pool' + name).output
            )
            block_outputs.append(
                base_model.get_layer('block4_pool' + name).output
            )
            block_outputs.append(
                base_model.get_layer('block5_pool' + name).output
            )
        elif model_name == 'resnet50':
            base_model = keras.applications.resnet50.ResNet50(include_top=include_top)
        else:
            print('please check the model name')
            assert False
        classifier_layer = tf.keras.layers.Dense(units=classes, activation=None, name='classifier'+name)

        logits = classifier_layer(tf.keras.layers.GlobalAveragePooling2D()(block_outputs[-1]))
        component_net = tf.keras.models.Model(org_input_tensor, logits, name='componentNet_' + name)
        if restore_path is not None:
            component_net.load_weights(restore_path)
        component_net.trainable = trainable
        return component_net, block_outputs, logits


def train_component_net(label1, label2):
    '''
    一般是训练一个binary的二分类任务
    :param label1
    :param label2
    '''
    x_train, y_train, x_test, y_test = return_part_mnist([label1, label2])
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))

    y_train[y_train == label1] = 0
    y_train[y_train == label2] = 1
    y_test[y_test == label1] = 0
    y_test[y_test == label2] = 1
    print(y_train[:128])
    print(y_test[:128])
    component_net, block_outputs, logits = build_base_model(name='{}_{}'.format(label1, label2), classes=2)
    print(component_net.inputs)
    component_net.compile(loss=keras.losses.sparse_categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(lr=0.00001),
                          metrics=['accuracy'])
    print(np.shape(x_train), np.shape(y_train))
    component_net.fit(x_train, y_train, batch_size=256, epochs=3, verbose=1)
    component_net.save_weights('./%d_%d.h5' % (label1, label2))
    print(np.shape(x_test), np.shape(y_test))
    score = component_net.evaluate(x_test, y_test, verbose=0, batch_size=256)
    print(score)


def cal_soft_loss(logits_t, logits_s):
    soft_loss = tf.keras.losses.mean_squared_error(logits_t, logits_s)
    return soft_loss


def cal_entropy(logits):
    softmax_layer = tf.keras.layers.Softmax()
    probs = softmax_layer(logits)
    entropy_layer = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.sum(-x * tf.log(tf.clip_by_value(x, 1e-8, 1.0)), axis=1))
    entropy = entropy_layer(probs)
    return entropy


def train_student_net(label1, label2, label3, label4):
    total_x_train = []
    total_y_train = []
    total_x_test = []
    total_y_test = []
    x_train, y_train, x_test, y_test = return_part_mnist([label1, label2])
    total_x_train.extend(x_train)
    total_y_train.extend(y_train)
    total_x_test.extend(x_test)
    total_y_test.extend(y_test)
    x_train, y_train, x_test, y_test = return_part_mnist([label3, label4])
    total_x_train.extend(x_train)
    total_y_train.extend(y_train)
    total_x_test.extend(x_test)
    total_y_test.extend(y_test)
    rnd_idxs = list(range(len(total_x_train)))
    np.random.shuffle(rnd_idxs)
    total_x_train = np.asarray(total_x_train, np.float)
    total_x_train = total_x_train[rnd_idxs]
    total_x_train = np.asarray(total_x_train, np.float)

    total_x_train = total_x_train[:-50]
    batch_size = 64
    train_ds = tf.data.Dataset.from_tensor_slices(total_x_train)
    train_ds = train_ds.batch(batch_size).repeat()  # batch 能给数据集增加批维度
    train_it = train_ds.make_one_shot_iterator()
    x_train_it = train_it.get_next()
    org_input_tensor = tf.keras.layers.Input(tensor=x_train_it)

    teacher_net1, t1_output_blocks, t1_logits = build_base_model('vgg16', include_top=False, classes=2, name='0_1',
                                                                 restore_path='./0_1.h5', trainable=False,
                                                                 org_input_tensor=org_input_tensor)
    teacher_net2, t2_output_blocks, t2_logits = build_base_model('vgg16', include_top=False, classes=2, name='2_3',
                                                                 restore_path='./2_3.h5', trainable=False,
                                                                 org_input_tensor=org_input_tensor)
    student_net, s_output_blocks, s_logits = build_base_model('vgg16', include_top=False, classes=4, name='0_1_2_3',
                                                              org_input_tensor=org_input_tensor)

    t1_entropy = cal_entropy(t1_logits)
    t2_entropy = cal_entropy(t2_logits)

    t1_probs = tf.nn.softmax(t1_logits)
    t1_pred_labels = tf.argmax(t1_probs, axis=1)
    t2_probs = tf.nn.softmax(t2_logits)
    t2_pred_labels = tf.argmax(t2_probs, axis=1)
    s_probs = tf.nn.softmax(s_logits)
    s_labels = tf.argmax(s_probs, axis=1)
    image_results = tf_put_text(org_input_tensor, t1_pred_labels, t2_pred_labels, s_labels, t1_entropy, t2_entropy)
    tf.summary.image('mnist/image', org_input_tensor, max_outputs=3)
    tf.summary.image('mnist/results', tf.cast(tf.expand_dims(image_results, axis=3), tf.uint8), max_outputs=3)

    t1_blocks = [FeatureAlignmentModule(block.get_shape().as_list()[-1]) for block in t1_output_blocks]
    t2_blocks = [FeatureAlignmentModule(block.get_shape().as_list()[-1]) for block in t2_output_blocks]
    t1_cb_outs = [t1_blocks[i].call(t1_output_blocks[i], s_output_blocks[i]) for i in range(len(t1_output_blocks))]
    t2_cb_outs = [t2_blocks[i].call(t2_output_blocks[i], s_output_blocks[i]) for i in range(len(t2_output_blocks))]
    t1_cb_losses = [FeatureAlignmentModule.block_loss(*t1_cb_outs[i]) for i in range(len(t1_cb_outs))]
    t2_cb_losses = [FeatureAlignmentModule.block_loss(*t2_cb_outs[i]) for i in range(len(t2_cb_outs))]
    for idx, t1_cb_loss in enumerate(t1_cb_losses):
        tf.summary.scalar('t1_block_loss/level'+str(idx), tf.reduce_mean(t1_cb_loss))
    for idx, t2_cb_loss in enumerate(t2_cb_losses):
        tf.summary.scalar('t2_block_loss/level_'+str(idx), tf.reduce_mean(t2_cb_loss))
    t1_cb_losses = tf.add_n(t1_cb_losses) / len(t1_cb_losses)
    t2_cb_losses = tf.add_n(t2_cb_losses) / len(t2_cb_losses)
    tf.summary.scalar('t1_block_loss/total', tf.reduce_mean(t1_cb_losses))
    tf.summary.scalar('t2_block_loss/total', tf.reduce_mean(t2_cb_losses))
    block_loss_layer = tf.keras.layers.Lambda(
        lambda paras: (lambda entropy1, entropy2,
                              losses1, losses2: tf.where(tf.keras.backend.less_equal(entropy1, entropy2),
                                                         losses1, losses2, name='cal_block_loss'))(*paras))
    block_loss_tensor = block_loss_layer((t1_entropy, t2_entropy, t1_cb_losses, t2_cb_losses))

    # 开始计算soft loss
    # print('t1_logits is ', t1_logits)
    # [5,2]
    new_t1_logits = tf.keras.backend.concatenate([t1_logits, tf.keras.backend.zeros_like(t1_logits)], axis=1)
    new_t2_logits = tf.keras.backend.concatenate(
        [tf.keras.backend.zeros_like(t2_logits), t2_logits], axis=1)

    soft_loss = tf.where_v2(tf.keras.backend.less_equal(t1_entropy, t2_entropy), cal_soft_loss(new_t1_logits, s_logits),
                            cal_soft_loss(new_t2_logits, s_logits))
    soft_loss = soft_loss * 200.
    print('soft loss is ', soft_loss)
    tf.summary.scalar('soft_loss', tf.reduce_mean(soft_loss))

    total_loss_layer = tf.keras.layers.Lambda(
        lambda x: (lambda xx, yy: tf.reduce_mean(xx + yy))(*x))
    total_loss_tensor = total_loss_layer((block_loss_tensor, soft_loss))

    target_model = tf.keras.Model(inputs=org_input_tensor, outputs=[t1_logits, t2_logits, s_logits])
    target_model.add_loss(total_loss_tensor)
    target_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001))
    tensorboard_callback = Tensorboard(summary_op=tf.summary.merge_all(), log_dir='./log/', batch_interval=5)
    target_model.fit(None, None, epochs=5, verbose=1, batch_size=batch_size, steps_per_epoch=24704/batch_size,
                     callbacks=[tensorboard_callback])
    target_model.save_weights('./0_1_2_3.h5')


def evaluate_component(label1, label2):
    org_input_tensor = tf.keras.layers.Input([28, 28, 1])
    student_net, s_output_blocks, s_logits = build_base_model('vgg16', include_top=False, classes=2,
                                                              name='{}_{}'.format(label1, label2),
                                                              restore_path='./{}_{}.h5'.format(label1, label2),
                                                              trainable=False, org_input_tensor=org_input_tensor)
    s_prob = tf.nn.softmax(s_logits)
    target_model = tf.keras.Model(inputs=org_input_tensor, outputs=s_prob)

    total_x_test = []
    total_y_test = []
    _, _, x_test, y_test = return_part_mnist([label1, label2])

    total_x_test.extend(x_test)
    total_y_test.extend(y_test)
    total_x_test = np.asarray(total_x_test, np.float)
    total_y_test = np.asarray(total_y_test)
    total_y_test[total_y_test == label1] = 0
    total_y_test[total_y_test == label2] = 1
    pred_res = target_model.predict(total_x_test, batch_size=64)
    print(np.argmax(pred_res, axis=1))
    print(total_y_test)
    print('the acc between {}, {} is'.format(label1, label2),
          (1.0 * np.sum(np.argmax(pred_res, axis=1) == total_y_test)) / len(total_y_test))


def evaluate():
    org_input_tensor = tf.keras.layers.Input([28, 28, 1])
    student_net, s_output_blocks, s_logits = build_base_model('vgg16', include_top=False, classes=4, name='0_1_2_3',
                                                              org_input_tensor=org_input_tensor)
    s_prob = tf.nn.softmax(s_logits)
    target_model = tf.keras.Model(inputs=org_input_tensor, outputs=s_prob)
    target_model.load_weights('./0_1_2_3.h5', by_name=True)

    total_x_test = []
    total_y_test = []
    _, _, x_test, y_test = return_part_mnist([0, 1])

    total_x_test.extend(x_test)
    total_y_test.extend(y_test)
    _, _, x_test, y_test = return_part_mnist([2, 3])

    total_x_test.extend(x_test)
    total_y_test.extend(y_test)
    total_x_test = np.asarray(total_x_test, np.float)
    total_y_test = np.asarray(total_y_test)
    pred_res = target_model.predict(total_x_test, batch_size=64)
    print(np.argmax(pred_res, axis=1), np.sum(np.argmax(pred_res, axis=1)))
    print(total_y_test)
    print('acc is ', (1.0 * np.sum(np.argmax(pred_res, axis=1) == total_y_test)) / len(total_y_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=int, default=1, help='the step ID, 1 or 2 or 3')
    args = vars(parser.parse_args())
    if args['step'] == 1:
        # 训练一个网络，可以分类0, 1
        train_component_net(0, 1)
        # 训练一个网络，可以分类2，3
        train_component_net(2, 3)
        evaluate_component(0, 1)
        evaluate_component(2, 3)
    elif args['step'] == 2:
        train_student_net(0, 1, 2, 3)
    elif args['step'] == 3:
        evaluate()
    else:
        assert False
