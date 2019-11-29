import tensorflow as tf
from seg_models.SegmentationModels import SegmentationModel
from datasets.Abdomen.parse_tfrecords import parse_tfrecords
from datasets.Abdomen import config as Abdomen_config
import os
from seg_models import config as Seg_config
import argparse
global args


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

    t1_model = SegmentationModel(args['t1_model_name'], args['t1_backbone_name'], 2, args['t1_activation'],
                                 batch_images_input, name=args['t1_name'])
    t2_model = SegmentationModel(args['t2_model_name'], args['t2_backbone_name'], 2, args['t2_activation'],
                                 batch_images_input, name=args['t2_name'])
    s_model = SegmentationModel(args['student_model_name'], args['student_backbone_name'], 2,
                                args['student_activation'], batch_images_input, name=args['student_name'])


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