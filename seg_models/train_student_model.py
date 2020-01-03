import sys
sys.path.append('..')
sys.path.append('.')
import tensorflow as tf
from seg_models.SegmentationModels import SegmentationModel, SegmentationClsModel
from datasets.Abdomen.parse_tfrecords import parse_tfrecords
from datasets.Abdomen import config as Abdomen_config
import os
from seg_models.config import TRAINING_STUDENT as training_config
import argparse
from modules import FeatureAlignmentModule
from callbacks import CustomCheckpointer, Tensorboard
global args
import segmentation_models as sm
sm.set_framework('tf.keras')


def compute_entropy(output, axis=-1, need_softmax=False):
    '''
    针对预测计算计算熵
    :param output: tensor
    :param axis: 计算熵的维度
    :param need_softmax: 是否需要做softmax
    :return:
    '''
    if need_softmax:
        softmax_layer = tf.keras.layers.Softmax()
        output = softmax_layer(output)
    entropy_layer = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.sum(-x * tf.log(tf.clip_by_value(x, 1e-8, 1.0)), axis=axis)
    )
    entropy = entropy_layer(output)
    return entropy


def non_selective_strategy(t1_prediction, t2_prediction):
    t1_prediction = tf.keras.backend.argmax(t1_prediction, axis=-1)
    t2_prediction = tf.keras.backend.argmax(t2_prediction, axis=-1)
    selective_mask = tf.keras.backend.zeros_like(t1_prediction)
    ones_mask = tf.keras.backend.ones_like(t1_prediction)
    selective_mask = tf.where(tf.equal(t1_prediction, 1), ones_mask, selective_mask)
    selective_mask = tf.where(tf.equal(t2_prediction, 1), tf.cast(ones_mask * 2, tf.int64), selective_mask)
    return selective_mask


def compute_organ_balanced_weights(t1_prediction, t2_prediction):
    '''
    根据liver和spleen的teacher 分割结果，分别得到权重矩阵，用于classification branch
    :param t1_prediction: [B, H, W, 2]
    :param t2_prediction: [B, H, W, 2]
    :return:
    '''
    t1_prediction = tf.keras.backend.argmax(t1_prediction, axis=-1)
    t2_prediction = tf.keras.backend.argmax(t2_prediction, axis=-1)
    size_t1 = tf.reduce_sum(t1_prediction, axis=[1, 2])
    size_t2 = tf.reduce_sum(t2_prediction, axis=[1, 2])
    zero_matrix = tf.zeros_like(size_t1, dtype=tf.float64)
    total_size = size_t1 + size_t2
    weight_t1 = tf.where(tf.equal(size_t1, 0), zero_matrix, total_size / size_t1)
    weight_t2 = tf.where(tf.equal(size_t2, 0), zero_matrix, total_size / size_t2)
    weight_t1 = tf.ones_like(t1_prediction, dtype=tf.float64) * weight_t1
    weight_t2 = tf.ones_like(t2_prediction, dtype=tf.float64) * weight_t2
    weight_matrix = tf.zeros_like(t1_prediction, dtype=tf.float64)
    weight_matrix = tf.where(tf.equal(t1_prediction, 1), weight_t1, weight_matrix)
    weight_matrix = tf.where(tf.equal(t2_prediction, 1), weight_t2, weight_matrix)
    return tf.cast(weight_matrix, tf.float32)


def selective_strategy(t1_prediction, t2_prediction, t1_entropy, t2_entropy):
    '''
    选择，返回mask tensor，针对每个像素的选择策略
    :param t1_prediction: [B, H, W, 2]
    :param t2_prediction: [B, H, W, 2]
    :param t1_entropy: [B, H, W]
    :param t2_entropy: [B, H, W]
    :return: [B, H, W]
    '''
    t1_prediction = tf.keras.backend.argmax(t1_prediction, axis=-1)
    t2_prediction = tf.keras.backend.argmax(t2_prediction, axis=-1)
    zeros_mask = tf.keras.backend.zeros_like(t1_prediction)
    ones_mask = tf.keras.backend.ones_like(t1_prediction)
    twos_mask = tf.keras.backend.ones_like(t1_prediction) * 2
    selective_mask = tf.keras.backend.zeros_like(t1_prediction)
    gen_mask = tf.keras.backend.zeros_like(t1_prediction)   # 生成的mask，由 0 1 2组成
    both_pos_mask = tf.where(tf.logical_and(tf.equal(t1_prediction, 1), tf.equal(t2_prediction, 1)), ones_mask,
                             zeros_mask)
    both_neg_mask = tf.where(tf.logical_and(tf.equal(t1_prediction, 0), tf.equal(t2_prediction, 0)), ones_mask,
                             zeros_mask)
    single_pos_mask = tf.where(tf.logical_xor(tf.equal(t1_prediction, 1), tf.equal(t2_prediction, 1)), ones_mask,
                               zeros_mask)
    t1_less_t2_mask = tf.where(tf.less_equal(t1_entropy, t2_entropy), ones_mask, twos_mask)

    # 将t1 预测为1，t2预测为0 的位置置为1
    selective_mask = tf.where(tf.logical_and(tf.equal(single_pos_mask, 1), tf.equal(t1_prediction, 1)), ones_mask,
                              selective_mask)

    # 将t1 预测为0，t2预测为1 的位置置为2
    selective_mask = tf.where(tf.logical_and(tf.equal(single_pos_mask, 1), tf.equal(t2_prediction, 1)), twos_mask,
                              selective_mask)
    # 将两个位置均为0的位置，t1的熵小于t2的位置置为1
    selective_mask = tf.where(tf.logical_and(tf.equal(both_neg_mask, 1), tf.equal(t1_less_t2_mask, 1)), ones_mask,
                              selective_mask)
    # 将两个位置均为0的位置，t1的熵大于t2的位置置为1
    selective_mask = tf.where(tf.logical_and(tf.equal(both_neg_mask, 1), tf.equal(t1_less_t2_mask, 2)), twos_mask,
                              selective_mask)
    # 将两个位置均为1的位置，t1的熵小于t2的位置置为1
    selective_mask = tf.where(tf.logical_and(tf.equal(both_pos_mask, 1), tf.equal(t1_less_t2_mask, 1)), ones_mask,
                              selective_mask)
    # 将两个位置均为1的位置，t1的熵大于t2的位置置为1
    selective_mask = tf.where(tf.logical_and(tf.equal(both_pos_mask, 1), tf.equal(t1_less_t2_mask, 2)), twos_mask,
                              selective_mask)

    gen_mask = tf.where(tf.logical_and(tf.equal(t1_prediction, 1), tf.equal(selective_mask, 1)), ones_mask, gen_mask)
    gen_mask = tf.where(tf.logical_and(tf.equal(t2_prediction, 1), tf.equal(selective_mask, 2)), twos_mask, gen_mask)
    return selective_mask, gen_mask


def compute_soft_loss(logits_t, logits_s):
    soft_loss = tf.keras.losses.mean_squared_error(logits_t, logits_s)
    return soft_loss


def main():
    global args
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    tf.keras.backend.set_session(sess)
    dataset_config = Abdomen_config.getDatasetConfigFactory('Abdomen')
    batch_images, batch_masks = parse_tfrecords(
        dataset_dir=os.path.join(dataset_config.RAW_DATA_TF_DIR, args['dataset_name']),
        batch_size=training_config['batch_size'], shuffle_size=training_config['shuffle_size'],
        prefetch_size=training_config['prefetch_size'])
    batch_images_input = tf.keras.layers.Input(tensor=batch_images)

    # build the basic model (student, teacher, teacher)
    t1_model = SegmentationModel(args['t1_model_name'], args['t1_backbone_name'], 2, args['t1_activation'],
                                 batch_images_input, name=args['t1_name'], trainable=False)
    t1_model.restore(args['t1_restore_path'])
    t2_model = SegmentationModel(args['t2_model_name'], args['t2_backbone_name'], 2, args['t2_activation'],
                                 batch_images_input, name=args['t2_name'], trainable=False)
    t2_model.restore(args['t2_restore_path'])
    if args['classification_flag']:
        s_model = SegmentationClsModel(args['student_model_name'], args['student_backbone_name'], 2,
                                args['student_activation'], input_tensor=batch_images_input, name=args['student_name'],
                                trainable=True)
        tf.summary.image('prediction/student_cls',
                         tf.expand_dims(tf.cast(tf.argmax(s_model.cls_prediction, axis=3) * 100, tf.uint8), axis=3),
                         max_outputs=3)
        tf.summary.image('prediction/student_seg', tf.expand_dims(s_model.prediction[:, :, :, 1], axis=3),
                         max_outputs=3)
        tf.summary.scalar('student/cls_prediction1_mean', tf.reduce_mean(s_model.cls_prediction[:, :, :, 1]))
        tf.summary.scalar('student/cls_prediction1_max', tf.reduce_max(s_model.cls_prediction[:, :, :, 1]))
        tf.summary.scalar('student/cls_prediction1_min', tf.reduce_min(s_model.cls_prediction[:, :, :, 1]))

    else:
        s_model = SegmentationModel(args['student_model_name'], args['student_backbone_name'], 3,
                                    args['student_activation'], input_tensor=batch_images,
                                    name=args['student_name'], trainable=True)
        tf.summary.image('prediction/student', tf.expand_dims(s_model.prediction[:, :, :, 1], axis=3), max_outputs=3)
    # add to tensorboard
    tf.summary.image('input', batch_images_input, max_outputs=3)
    tf.summary.image('prediction/t1', tf.cast(tf.expand_dims(t1_model.prediction[:, :, :, 1], axis=3) * 200, tf.uint8),
                     max_outputs=3)
    tf.summary.image('prediction/t2', tf.cast(tf.expand_dims(t2_model.prediction[:, :, :, 1], axis=3) * 200, tf.uint8),
                     max_outputs=3)
    tf.summary.image('gt', tf.expand_dims(tf.cast(batch_masks * 100, tf.uint8), axis=3), max_outputs=3)

    # build the block loss
    t1_fa_blocks = [FeatureAlignmentModule(encoder_output.get_shape().as_list()[-1] // 2, require_resize=True) for
                    encoder_output in t1_model.encoder_outputs]
    t2_fa_blocks = [FeatureAlignmentModule(encoder_output.get_shape().as_list()[-1] // 2, require_resize=True) for
                    encoder_output in t2_model.encoder_outputs]
    t1_fa_blocks_outputs = [t1_fa_blocks[i].call(t1_model.encoder_outputs[i], s_model.encoder_outputs[i]) for i in
                            range(len(t1_model.encoder_outputs))]
    t2_fa_blocks_outputs = [t2_fa_blocks[i].call(t2_model.encoder_outputs[i], s_model.encoder_outputs[i]) for i in
                            range(len(t2_model.encoder_outputs))]
    # the element is matrix B 512, 512
    t1_s_blocks_losses = [FeatureAlignmentModule.block_loss(*t1_fa_blocks_outputs[i], pixel_wise=True) for i in
                          range(len(t1_model.encoder_outputs))]
    t2_s_blocks_losses = [FeatureAlignmentModule.block_loss(*t2_fa_blocks_outputs[i], pixel_wise=True) for i in
                          range(len(t1_model.encoder_outputs))]
    print('t1_s_blocks_losses are ', t1_s_blocks_losses)
    print('t2_s_blocks_losses are ', t2_s_blocks_losses)
    for i in range(len(t1_model.encoder_outputs)):
        tf.summary.scalar('t1_blocks_losses/level_{}'.format(i), tf.reduce_mean(t1_s_blocks_losses[i]))
        tf.summary.scalar('t2_blocks_losses/level_{}'.format(i), tf.reduce_mean(t2_s_blocks_losses[i]))
    t1_s_block_loss = tf.add_n(t1_s_blocks_losses) / (1.0 * len(t1_model.encoder_outputs))
    t2_s_block_loss = tf.add_n(t2_s_blocks_losses) / (1.0 * len(t1_model.encoder_outputs))
    tf.summary.scalar('t1_blocks_losses/average', tf.reduce_mean(t1_s_block_loss))
    tf.summary.scalar('t2_blocks_losses/average', tf.reduce_mean(t2_s_block_loss))

    if args['selective_flag']:
        # build the selective mask tensor
        get_entropy_layer = tf.keras.layers.Lambda(lambda x: compute_entropy(x))
        t1_entropy = get_entropy_layer(t1_model.prediction)
        t2_entropy = get_entropy_layer(t2_model.prediction)
        selective_mask, gen_mask = selective_strategy(t1_model.prediction, t2_model.prediction, t1_entropy, t2_entropy)
        print('selective_mask is ', selective_mask)
        tf.summary.image('selective_mask', tf.cast(tf.expand_dims(selective_mask * 100, axis=3), tf.uint8),
                         max_outputs=3)
        tf.summary.image('gen_mask', tf.cast(tf.expand_dims(gen_mask * 100, axis=3), tf.uint8),
                         max_outputs=3)
        # build the last logits loss
        get_soft_loss_layer = tf.keras.layers.Lambda(lambda x: (lambda t, s: compute_soft_loss(t, s))(*x))
        print(t1_model.decoder_outputs[-1], s_model.decoder_outputs[-1])
        t1_soft_loss = get_soft_loss_layer((t1_model.decoder_outputs[-1], s_model.decoder_outputs[-1]))
        t2_soft_loss = get_soft_loss_layer((t2_model.decoder_outputs[-1], s_model.decoder_outputs[-1]))
        print('t1_soft_loss is ', t1_soft_loss)
        print('t2_soft_loss is ', t2_soft_loss)
        tf.summary.scalar('soft_loss/t1', tf.reduce_mean(t1_soft_loss))
        tf.summary.scalar('soft_loss/t2', tf.reduce_mean(t2_soft_loss))
        # build the total loss for teacher1 or teacher2
        soft_coff = 1.0
        block_coff = 0.0
        t1_loss = block_coff * t1_s_block_loss + t1_soft_loss * soft_coff
        t2_loss = block_coff * t2_s_block_loss + t2_soft_loss * soft_coff

        # build the selective loss
        selective_loss = tf.zeros_like(selective_mask, dtype=tf.float32)
        print(selective_mask, selective_loss, t1_loss, t2_loss)
        selective_loss = tf.where(tf.equal(selective_mask, 1), t1_loss, selective_loss)
        selective_loss = tf.where(tf.equal(selective_mask, 2), t2_loss, selective_loss)

        if args['classification_flag']:
            lambda_cls_loss = 10.0
            organ_balanced_weight = compute_organ_balanced_weights(t1_prediction=t1_model.prediction,
                                                                   t2_prediction=t2_model.prediction)
            cls_loss = tf.keras.losses.sparse_categorical_crossentropy(gen_mask, s_model.cls_prediction)
            cls_loss = cls_loss * (1 + organ_balanced_weight)
            tf.summary.scalar('loss/selective_loss', tf.reduce_mean(selective_loss))
            selective_loss = tf.reduce_mean(selective_loss) + tf.reduce_mean(cls_loss) * lambda_cls_loss
            tf.summary.scalar('loss/cls_loss', tf.reduce_mean(cls_loss))

    else:
        gen_gt_mask = non_selective_strategy(t1_prediction=t1_model.prediction, t2_prediction=t2_model.prediction)
        tf.summary.image('gen_mask', tf.cast(tf.expand_dims(gen_gt_mask * 100, axis=3), tf.uint8),
                         max_outputs=3)
        selective_loss = tf.keras.losses.sparse_categorical_crossentropy(gen_gt_mask, s_model.prediction)
        tf.summary.image('selective_mask', tf.cast(tf.expand_dims(gen_gt_mask * 100, axis=3), tf.uint8),
                         max_outputs=3)

    # build the trained model
    if args['classification_flag']:
        print('do classification')
        trained_model = tf.keras.models.Model(inputs=batch_images_input,
                                              outputs=[t1_model.prediction, t2_model.prediction, s_model.prediction,
                                                       s_model.cls_prediction])
    else:
        print('non classification')
        trained_model = tf.keras.models.Model(inputs=batch_images_input,
                                              outputs=[t1_model.prediction, t2_model.prediction, s_model.prediction])
    trained_model.add_loss(selective_loss)
    trained_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    tensorboard_callback = Tensorboard(summary_op=tf.summary.merge_all(), log_dir='./log/', batch_interval=10,
                                       batch_size=training_config['batch_size'],
                                       size_dataset=Abdomen_config.get_dataset_config(args['dataset_name'])['size'])
    cb_checkpointer = CustomCheckpointer(os.path.join(args['save_dir'], args['dataset_name']), trained_model,
                                         monitor='loss',
                                         mode='min', save_best_only=False, verbose=1,
                                         save_model=False,
                                         prefix='{}-{}-{}'.format(args['student_name'], args['student_model_name'],
                                                                  args['student_backbone_name']) if args[
                                             'selective_flag'] else '{}-{}-{}-False'.format(args['student_name'],
                                                                                            args['student_model_name'],
                                                                                            args[
                                                                                                'student_backbone_name']))
    trained_model.fit(epochs=args['num_epoches'],
                      steps_per_epoch=Abdomen_config.get_dataset_config(args['dataset_name'])['size'] //
                                      training_config['batch_size'], callbacks=[tensorboard_callback, cb_checkpointer])


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t1_m', '--t1_model_name', type=str, default='unet',
                        help='the network student used')
    parser.add_argument('-t1_b', '--t1_backbone_name', type=str, default='vgg',
                        help='the network student used')
    parser.add_argument('-t1_r', '--t1_restore_path', type=str,
                        default='/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/'
                                'liver-unet-vgg-ep49-End-loss0.0015.h5',
                        help='')
    parser.add_argument('-t1_n', '--t1_name', type=str, default='liver', help='')
    parser.add_argument('-t1_a', '--t1_activation', type=str, default='softmax',
                        help='the dataset dir which storage tfrecords files ')

    parser.add_argument('-t2_m', '--t2_model_name', type=str, default='unet',
                        help='the network student used')
    parser.add_argument('-t2_b', '--t2_backbone_name', type=str, default='vgg',
                        help='the network student used')
    parser.add_argument('-t2_r', '--t2_restore_path', type=str,
                        default='/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/'
                                'spleen-unet-vgg-ep49-End-loss0.0006.h5',
                        help='')
    parser.add_argument('-t2_n', '--t2_name', type=str, default='spleen', help='')
    parser.add_argument('-t2_a', '--t2_activation', type=str, default='softmax',
                        help='the dataset dir which storage tfrecords files ')

    parser.add_argument('-s_m', '--student_model_name', type=str, default='unet',
                        help='the network student used')
    parser.add_argument('-s_b', '--student_backbone_name', type=str, default='vgg',
                        help='the network student used')
    parser.add_argument('-s_a', '--student_activation', type=str, default='softmax',
                        help='the dataset dir which storage tfrecords files ')
    parser.add_argument('-s_name', '--student_name', type=str, default='liver_spleen',
                        help='the network student used')

    parser.add_argument('-d_m', '--dataset_name', type=str, default='V1',
                        help='the name of dataset')

    parser.add_argument('-s', '--save_dir', type=str,
                        default='/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/')
    parser.add_argument('-n_e', '--num_epoches', type=int, default=50,
                        help='')
    parser.add_argument('-s_f', '--selective_flag', action='store_true', default=False,
                        help='')
    parser.add_argument('-c_f', '--classification_flag', action='store_true', default=False,
                        help='')
    args = vars(parser.parse_args())
    main()