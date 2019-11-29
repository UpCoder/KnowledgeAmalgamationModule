import tensorflow as tf
from seg_models.SegmentationModels import SegmentationModel
import os
from datasets.medical_image_utils import read_nii, save_mhd_image
from datasets.Abdomen import config
import numpy as np


class Utils:
    @staticmethod
    def binary_dice(gt, pred):
        intersection = np.sum(np.logical_and(gt == 1, pred == 1))
        union = np.sum(gt == 1) + np.sum(pred == 1)
        print('intersection is ', intersection)
        print(np.sum(gt == 1), np.sum(pred == 1))
        return 2.0 * intersection / union


def evaluate_teacher():
    restore_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/' \
                   'liver-unet-vgg-ep49-End-loss0.0015.h5'
    base_name = os.path.basename(restore_path)
    target_name, model_name, backbone_name = base_name.split('-')[:3]
    input_layer = tf.keras.layers.Input(shape=(512, 512, 3))
    seg_model = SegmentationModel(model_name, backbone_name, 2, name=target_name, input_tensor=input_layer)
    seg_model.restore(restore_path)
    dices = []
    for idx in config.EVALUATING_RANGE:
        print(idx)
        case_name = '{:04d}.nii'.format(idx)
        img_path = os.path.join(config.RAW_DATA_EVALUATING_DIR, 'img', 'img' + case_name)
        label_path = os.path.join(config.RAW_DATA_TRAINING_DIR, 'label',
                                  'label' + case_name)
        img = read_nii(img_path)
        label = read_nii(label_path)
        zero_matrixs = np.zeros_like(label)
        zero_matrixs[label == config.name2global_label[target_name]] = 1
        label = zero_matrixs

        window_center = config.get_dataset_config('V1')['window_center']
        window_width = config.get_dataset_config('V1')['window_width']
        window_left = window_center - window_width / 2
        window_right = window_center + window_width / 2
        img[img < window_left] = window_left
        img[img > window_right] = window_right

        img = np.transpose(img, axes=[2, 0, 1])
        label = np.transpose(label, axes=[2, 0, 1])
        img = np.expand_dims(img, axis=3)
        img = np.concatenate([img, img, img], axis=-1)
        img = np.asarray(img, np.float)

        prediction_probs = seg_model.predict(img, batch_size=8)
        prediction = np.argmax(prediction_probs, axis=-1)
        save_nii_path = os.path.join(config.RAW_DATA_EVALUATING_DIR, 'pred', 'pred_' + case_name[:-4] + '.mhd')
        save_mhd_image(np.transpose(prediction, axes=[0, 2, 1]), save_nii_path)
        print(np.shape(img), np.shape(label), np.shape(prediction))
        print(np.max(label), np.min(label))
        print(np.max(prediction), np.min(prediction))
        dice = Utils.binary_dice(label, prediction)

        print('{}: {}'.format(case_name, dice))
        dices.append(dice)
    print('global average dice is ', np.mean(dices))


if __name__ == '__main__':
    evaluate_teacher()