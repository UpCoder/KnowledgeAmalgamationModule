import sys
sys.path.append('..')
sys.path.append('.')
import tensorflow as tf
from datasets.Abdomen.parse_tfrecords import parse_tfrecords
from seg_models.SegmentationModels import SegmentationModel
from datasets.Abdomen import config as Abdomen_config
from seg_models.config import TRAINING as training_config
import argparse
import os
from callbacks import Tensorboard, CustomCheckpointer
global args
# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=kill -9


def main():
    global args
    # configuration about gpu
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    tf.keras.backend.set_session(sess)

    dataset_config = Abdomen_config.getDatasetConfigFactory(args['dataset_name'])
    # prepare dataset
    batch_images, batch_masks = parse_tfrecords(
        dataset_dir=os.path.join(dataset_config.RAW_DATA_TF_DIR, args['dataset_version']),
        batch_size=training_config['batch_size'], shuffle_size=training_config['shuffle_size'],
        prefetch_size=training_config['prefetch_size'])
    batch_images_input = tf.keras.layers.Input(tensor=batch_images)
    batch_masks = tf.keras.backend.cast(tf.keras.backend.equal(batch_masks, args['target_label']), tf.int32)

    # put the input to segmentation model
    seg_model = SegmentationModel(args['model_name'], args['backbone_name'], 2, args['activation'], batch_images_input,
                                  name=args['target_name'])
    print(batch_images_input, seg_model.prediction, batch_masks)

    # add to tensorboard
    tf.summary.image('input', batch_images_input, max_outputs=3)
    tf.summary.image('prediction', tf.expand_dims(seg_model.prediction[:, :, :, 1], axis=3), max_outputs=3)
    tf.summary.image('gt', tf.expand_dims(tf.cast(batch_masks * 200, tf.uint8), axis=3), max_outputs=3)

    # calculate the loss
    final_loss = seg_model.build_loss(batch_masks, cross_entropy_coff=args['cross_entropy'], dice_coff=args['dice'],
                                      focal_loss_coff=args['focal_loss'], triplet_loss_coff=args['triplet_loss'])

    # build the trained model
    trained_model = tf.keras.Model(inputs=batch_images_input, outputs=seg_model.prediction, name='final_model')

    # whether to restore model
    if args['restore_path'] is not None:
        trained_model.load_weights(args['restore_path'])

    # configuration about the trained model
    trained_model.add_loss(final_loss)
    trained_model.compile(tf.keras.optimizers.Adam(lr=1e-4))

    # callbacks
    tensorboard_callback = Tensorboard(summary_op=tf.summary.merge_all(), log_dir='./log/', batch_interval=10,
                                       batch_size=training_config['batch_size'],
                                       size_dataset=Abdomen_config.get_dataset_config(args['dataset_version'])['size'])
    cb_checkpointer = CustomCheckpointer(os.path.join(args['save_dir'], args['dataset_version']), seg_model.seg_model, monitor='loss',
                                         mode='min', save_best_only=False, verbose=1,
                                         prefix='{}-{}-{}'.format(args['target_name'], args['model_name'],
                                                                  args['backbone_name']))
    trained_model.fit(epochs=args['num_epoches'],
                      steps_per_epoch=Abdomen_config.get_dataset_config(args['dataset_version'])['size'] //
                                      training_config['batch_size'], callbacks=[tensorboard_callback, cb_checkpointer])


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='unet',
                        help='the network student used')
    parser.add_argument('-b', '--backbone_name', type=str, default='vgg',
                        help='the network student used')
    parser.add_argument('-d_n', '--dataset_name', type=str, default='Abdomen',
                        help='the name of dataset')
    parser.add_argument('-d_v', '--dataset_version', type=str, default='V2',
                        help='the name of dataset')
    parser.add_argument('-a', '--activation', type=str, default='softmax',
                        help='the dataset dir which storage tfrecords files ')
    parser.add_argument('-s', '--save_dir', type=str,
                        default='/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck2/')
    parser.add_argument('-t_n', '--target_name', type=str, default='liver',
                        help='spleen / liver')
    parser.add_argument('-t_l', '--target_label', type=int, default=2,
                        help='1 / 2')
    parser.add_argument('-n_e', '--num_epoches', type=int, default=50,
                        help='1 / 2')
    parser.add_argument('-r', '--restore_path', type=str,
                        default=None
                        # default='/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/'
                        #         'liver-unet-vgg-ep10-End-loss0.0060.tf'
                        )
    parser.add_argument('-l_c', '--cross_entropy', type=float, default=1.0)
    parser.add_argument('-l_d', '--dice', type=float, default=0.0)
    parser.add_argument('-l_f', '--focal_loss', type=float, default=0.0)
    parser.add_argument('-l_t', '--triplet_loss', type=float, default=0.0)
    args = vars(parser.parse_args())
    main()