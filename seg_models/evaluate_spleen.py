import os
import tensorflow as tf
from seg_models.SegmentationModels import SegmentationModel
from datasets.Abdomen.nii2PNG import get_nii2npy_func
from datasets.medical_image_utils import save_mhd_image
import numpy as np
from datasets.Abdomen import config
import cv2
gpu_config = tf.ConfigProto(allow_soft_placement=True)
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
tf.keras.backend.set_session(sess)


class Utils:
    @staticmethod
    def binary_dice(gt, pred, target_label=1):
        intersection = np.sum(np.logical_and(gt == 1, pred == target_label))
        union = np.sum(gt == 1) + np.sum(pred == target_label)
        # print('intersection is ', intersection)
        # print(np.sum(gt == 1), np.sum(pred == 1))
        return 2.0 * intersection / union


def evaluate_single_teacher(restore_path=None):
    '''
    验证单独的spleen 数据集
    :param restore_path:
    :return:
    '''
    dataset_config = config.getDatasetConfigFactory('chen_spleen')
    if restore_path is None:
        # restore_path = '/media/give/HDD3/ld/Documents/datasets/chen_spleen/ck/V2/' \
        #                'spleen-unet-resnet-ep35-End-loss0.00251.h5' # 0.95
        restore_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck2/V2/' \
                       'spleen_liver-unet-resnet-ep99-End-loss0.0007.h5'
    version_name = os.path.basename(os.path.dirname(restore_path))
    base_name = os.path.basename(restore_path)
    target_name, model_name, backbone_name = base_name.split('-')[:3]
    print(target_name, str(target_name).count('_'))
    num_classes = str(target_name).count('_') + 2
    target_label = 1 if str(target_name).count('_') >= 1 else 1
    input_layer = tf.keras.layers.Input(shape=(512, 512, 3))
    seg_model = SegmentationModel(model_name, backbone_name, num_classes, name=target_name, input_tensor=input_layer)
    seg_model.restore(restore_path)
    dices = []
    predictions = []
    # for idx in [7]:
    for idx in range(1000, 1055):
        print(idx)
        case_name = '{:04d}.nii'.format(idx)
        img, label = get_nii2npy_func(version_name)(case_name, is_training=False, img_prefix='volume-ID_',
                                                    label_prefix='segmentation-ID_',
                                                    Training_DIR=dataset_config.RAW_DATA_EVALUATING_DIR)
        if img is None:
            print('skip ', case_name)
            continue

        b, w, h, c = np.shape(img)
        img = np.asarray(img, np.float)
        label = np.asarray(label, np.int)
        if w != 512 or h != 512:
            resized_img = np.zeros([b, 512, 512, c], np.float)
            resized_label = np.zeros([b, 512, 512], np.float)
            for slice_idx, (img_slice, label_slice) in enumerate(zip(img, label)):
                img_slice = cv2.resize(img_slice, (512, 512))
                label_slice = cv2.resize(label_slice, (512, 512), interpolation=cv2.INTER_NEAREST)
                resized_img[slice_idx] = img_slice
                resized_label[slice_idx] = label_slice
            img = resized_img
            label = resized_label
        prediction_probs = seg_model.predict(img, batch_size=8)
        prediction = np.argmax(prediction_probs, axis=-1)
        save_nii_path = os.path.join(dataset_config.RAW_DATA_EVALUATING_DIR, 'pred',
                                     'pred_' + target_name + '_' + case_name[:-4] + '.mhd')
        print(save_nii_path)
        save_mhd_image(np.transpose(prediction, axes=[0, 2, 1]), save_nii_path)
        print(np.shape(img), np.shape(label), np.shape(prediction))
        print(np.max(label), np.min(label))
        print(np.max(prediction), np.min(prediction))
        dice = Utils.binary_dice(label, prediction, target_label=target_label)
        predictions.append(prediction)
        print('{}: {}'.format(case_name, dice))
        dices.append(dice)
    print('global average dice is ', np.mean(dices))
    print('dices are ', dices)
    return predictions


if __name__ == '__main__':
    evaluate_single_teacher()