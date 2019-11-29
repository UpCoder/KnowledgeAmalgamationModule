import tensorflow as tf
from seg_models.SegmentationModels import SegmentationModel
from datasets.Abdomen.parse_tfrecords import parse_tfrecords
from datasets.Abdomen import config as Abdomen_config
import os
from seg_models import config as Seg_config
import argparse
from modules import FeatureAlignmentModule
global args


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
    both_pos_mask = tf.where(tf.logical_and(tf.equal(t1_prediction, 1), tf.equal(t1_prediction, 1)), ones_mask,
                             zeros_mask)
    both_neg_mask = tf.where(tf.logical_and(tf.equal(t1_prediction, 0), tf.equal(t2_prediction, 0)), ones_mask,
                             zeros_mask)
    single_pos_mask = tf.where(tf.logical_xor(tf.equal(t1_prediction, 1), tf.equal(t2_prediction, 1)), ones_mask,
                               zeros_mask)
    t1_less_t2_mask = tf.where(tf.less_equal(t1_entropy, t2_entropy), ones_mask, twos_mask)

    selective_mask = tf.where(tf.logical_and(single_pos_mask, tf.equal(t1_prediction, 1)), ones_mask, selective_mask)
    selective_mask = tf.where(tf.logical_and(single_pos_mask, tf.equal(t2_prediction, 1)), twos_mask, selective_mask)
    selective_mask = tf.where(tf.logical_and(both_neg_mask, t1_less_t2_mask), ones_mask, selective_mask)
    selective_mask = tf.where(tf.logical_and(both_neg_mask, tf.logical_not(t1_less_t2_mask)), twos_mask, selective_mask)
    selective_mask = tf.where(tf.logical_and(both_pos_mask, t1_less_t2_mask), ones_mask, selective_mask)
    selective_mask = tf.where(tf.logical_and(both_pos_mask, tf.logical_not(t1_less_t2_mask)), twos_mask, selective_mask)
    return selective_mask


def compute_soft_loss(logits_t, logits_s):
    soft_loss = tf.keras.losses.mean_squared_error(logits_t, logits_s)
    return soft_loss


def main():
    global args
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    tf.keras.backend.set_session(sess)

    batch_images, batch_masks = parse_tfrecords(
        dataset_dir=os.path.join(Abdomen_config.RAW_DATA_TF_DIR, args['dataset_name']),
        batch_size=Seg_config.TRAINING['batch_size'], shuffle_size=Seg_config.TRAINING['shuffle_size'],
        prefetch_size=Seg_config.TRAINING['prefetch_size'])
    batch_images_input = tf.keras.layers.Input(tensor=batch_images)

    # build the basic model (student, teacher, teacher)
    t1_model = SegmentationModel(args['t1_model_name'], args['t1_backbone_name'], 2, args['t1_activation'],
                                 batch_images_input, name=args['t1_name'], trainable=False)
    t1_model.restore(args['t1_restore_path'])
    t2_model = SegmentationModel(args['t2_model_name'], args['t2_backbone_name'], 2, args['t2_activation'],
                                 batch_images_input, name=args['t2_name'], trainable=False)
    t2_model.restore(args['t2_restore_path'])
    s_model = SegmentationModel(args['student_model_name'], args['student_backbone_name'], 2,
                                args['student_activation'], batch_images_input, name=args['student_name'])

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
    t1_s_block_loss = [FeatureAlignmentModule.block_loss(*t1_fa_blocks_outputs[i], pixel_wise=True) for i in
                       range(len(t1_model.encoder_outputs))]
    t2_s_block_loss = [FeatureAlignmentModule.block_loss(*t2_fa_blocks_outputs[i], pixel_wise=True) for i in
                       range(len(t1_model.encoder_outputs))]

    get_entropy_layer = tf.keras.layers.Lambda(lambda x: compute_entropy(x))
    t1_entropy = get_entropy_layer(t1_model.prediction)
    t2_entropy = get_entropy_layer(t2_model.prediction)
    selective_mask = selective_strategy(t1_model.prediction, t2_model.prediction, t1_entropy, t2_entropy)
    tf.summary.image('selective_mask', tf.expand_dims(selective_mask * 100, axis=3), max_outputs=3)

    get_soft_loss_layer = tf.keras.layers.Lambda(lambda x: (lambda t, s: compute_entropy(t, s))(*x))
    t1_soft_loss = get_soft_loss_layer(t1_model.decoder_outputs[-1], s_model.decoder_outputs[-1])
    t2_soft_loss = get_soft_loss_layer(t2_model.decoder_outputs[-1], s_model.decoder_outputs[-1])

    trained_model = tf.keras.models.Model(inputs=batch_images_input,
                                          outputs=[t1_model.prediction, t2_model.prediction, s_model.prediction])


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t1_m', '--t1_model_name', type=str, default='unet',
                        help='the network student used')
    parser.add_argument('-t1_b', '--t1_backbone_name', type=str, default='vgg',
                        help='the network student used')
    parser.add_argument('-t1_r', '--t1_restore_path', type=str, default=None, help='')
    parser.add_argument('-t1_n', '--t1_name', type=str, default='liver', help='')
    parser.add_argument('-t1_a', '--t1_activation', type=str, default='softmax',
                        help='the dataset dir which storage tfrecords files ')

    parser.add_argument('-t2_m', '--t2_model_name', type=str, default='unet',
                        help='the network student used')
    parser.add_argument('-t2_b', '--t2_backbone_name', type=str, default='vgg',
                        help='the network student used')
    parser.add_argument('-t2_r', '--t2_restore_path', type=str, default=None, help='')
    parser.add_argument('-t2_n', '--t2_name', type=str, default='spleen', help='')
    parser.add_argument('-t2_a', '--t2_activation', type=str, default='softmax',
                        help='the dataset dir which storage tfrecords files ')

    parser.add_argument('-s_m', '--student_model_name', type=str, default='unet',
                        help='the network student used')
    parser.add_argument('-s_b', '--student_backbone_name', type=str, default='vgg',
                        help='the network student used')
    parser.add_argument('-s_a', '--student_activation', type=str, default='softmax',
                        help='the dataset dir which storage tfrecords files ')

    parser.add_argument('-d_m', '--dataset_name', type=str, default='V1',
                        help='the name of dataset')

    parser.add_argument('-s', '--save_dir', type=str,
                        default='/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/')
    parser.add_argument('-n_e', '--num_epoches', type=int, default=50,
                        help='')
    args = vars(parser.parse_args())