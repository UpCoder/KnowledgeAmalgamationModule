import tensorflow as tf
from seg_models.SegmentationModels import SegmentationModel
import os
from datasets.medical_image_utils import read_nii, save_mhd_image
from datasets.Abdomen import config
from datasets.Abdomen.nii2PNG import get_nii2npy_func
import numpy as np

gpu_config = tf.ConfigProto(allow_soft_placement=True)
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
tf.keras.backend.set_session(sess)


class Utils:
    @staticmethod
    def binary_dice(gt, pred):
        intersection = np.sum(np.logical_and(gt == 1, pred == 1))
        union = np.sum(gt == 1) + np.sum(pred == 1)
        print('intersection is ', intersection)
        print(np.sum(gt == 1), np.sum(pred == 1))
        return 2.0 * intersection / union


def evaluate_dual_teacher():
    # global average dice is 0.8778939842862805
    # dices are[ 0.9422165450939527, 0.906866104347195, 0.903115059832696, 0.9065054097843166,
    # 0.955396536785528, 0.757845242446916, 0.7864991558076108, 0.873478655351284, 0.8430822023083274,
    # 0.9039349311049786]
    liver_restore_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/' \
                         'liver-unet-vgg-ep49-End-loss0.0015.h5'
    spleen_restore_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/' \
                          'spleen-unet-vgg-ep49-End-loss0.0006.h5'
    liver_predictions = evaluate_single_teacher(liver_restore_path)
    spleen_predictions = evaluate_single_teacher(spleen_restore_path)
    target_name_list = ['liver', 'spleen']
    dices = []
    merged_predictions = []
    for idx, liver_prediction, spleen_prediction in zip(config.EVALUATING_RANGE, liver_predictions, spleen_predictions):
        merged_prediction = np.asarray((liver_prediction + spleen_prediction) >= 1, np.int)
        merged_predictions.append(merged_prediction)
        case_name = '{:04d}.nii'.format(idx)
        label_path = os.path.join(config.RAW_DATA_TRAINING_DIR, 'label',
                                  'label' + case_name)
        label = read_nii(label_path)
        label_matrixs = []
        all_label_matrix = np.zeros_like(label)
        for organ_name in target_name_list:
            all_label_matrix[label == config.name2global_label[organ_name]] = 1
            zero_matrixs = np.zeros_like(label)
            zero_matrixs[label == config.name2global_label[organ_name]] = 1
            label_matrixs.append(zero_matrixs)
        label = np.transpose(all_label_matrix, axes=[2, 0, 1])
        dice = Utils.binary_dice(label, merged_prediction)

        print('{}: {}'.format(case_name, dice))
        dices.append(dice)
    print('global average dice is ', np.mean(dices))
    print('dices are ', dices)
    return merged_predictions


def evaluate_single_teacher(restore_path=None):
    # liver
    # global average dice is 0.883625705306539
    # dices are[ 0.9647435018456458, 0.9031160523095303, 0.9066215911754599, 0.9109800871751279,
    # 0.9565866647475801, 0.7793411471113311, 0.7975616072483468, 0.9466400450570951,
    # 0.7669447817627334, 0.9037215746325405]

    # spleen
    # global average dice is 0.6781564219198277
    # dices are[ 0.7957785038850702, 0.9481434961485716, 0.8796940616424858, 0.8716361373837298,
    # 0.9481954867959325, 0.08366672464776483, 0.7120624859470861, 0.016120085278726434,
    # 0.6174779768437731, 0.9087892606251363]

    # spleen
    # V2/spleen-unet-vgg-ep99-End-loss0.0002.h5
    # global average dice is  0.8254234068246822
    # dices are  [0.9466422354146996, 0.961049902786779, 0.9228719372571547, 0.9396556424721599, 0.9296169181016685,
    #  0.7467543800765255, 0.43427207329268686, 0.9059204083925962, 0.5519578801164976, 0.9154926903360546]

    # global average dice is  0.8892064336207314
    # V2/liver-unet-vgg-ep99-End-loss0.0009.h5
    # dices are  [0.9675529257852, 0.971656139315968, 0.9254148599361137, 0.9390645828817339, 0.9320660538743973,
    # 0.9161950014051498, 0.3812259390300064, 0.9565657718401224, 0.9461939440687256, 0.9561291180698972]
    if restore_path is None:
        restore_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V2/' \
                       'liver-unet-vgg-ep50-End-loss0.0017.h5'
    version_name = os.path.basename(os.path.dirname(restore_path))
    base_name = os.path.basename(restore_path)
    target_name, model_name, backbone_name = base_name.split('-')[:3]
    input_layer = tf.keras.layers.Input(shape=(512, 512, 3))
    seg_model = SegmentationModel(model_name, backbone_name, 2, name=target_name, input_tensor=input_layer)
    seg_model.restore(restore_path)
    dices = []
    predictions = []
    # for idx in [7]:
    for idx in config.EVALUATING_RANGE:
        print(idx)
        case_name = '{:04d}.nii'.format(idx)
        # img_path = os.path.join(config.RAW_DATA_EVALUATING_DIR, 'img', 'img' + case_name)
        # label_path = os.path.join(config.RAW_DATA_TRAINING_DIR, 'label',
        #                           'label' + case_name)
        # img = read_nii(img_path)
        # label = read_nii(label_path)

        #
        # window_center = config.get_dataset_config('V1')['window_center']
        # window_width = config.get_dataset_config('V1')['window_width']
        # window_left = window_center - window_width / 2
        # window_right = window_center + window_width / 2
        # img[img < window_left] = window_left
        # img[img > window_right] = window_right
        img, label = get_nii2npy_func(version_name)(case_name, is_training=False)
        zero_matrixs = np.zeros_like(label)
        zero_matrixs[label == config.name2global_label[target_name]] = 1
        label = zero_matrixs
        prediction_probs = seg_model.predict(img, batch_size=8)
        prediction = np.argmax(prediction_probs, axis=-1)
        save_nii_path = os.path.join(config.RAW_DATA_EVALUATING_DIR, 'pred',
                                     'pred_' + target_name + '_' + case_name[:-4] + '.mhd')
        save_mhd_image(np.transpose(prediction, axes=[0, 2, 1]), save_nii_path)
        print(np.shape(img), np.shape(label), np.shape(prediction))
        print(np.max(label), np.min(label))
        print(np.max(prediction), np.min(prediction))
        dice = Utils.binary_dice(label, prediction)
        predictions.append(prediction)
        print('{}: {}'.format(case_name, dice))
        dices.append(dice)
    print('global average dice is ', np.mean(dices))
    print('dices are ', dices)
    return predictions


def evaluate_student():
    # global average dice is 0.8906522556217844
    # dices are[ 0.9385382739727629, 0.9361881981945765, 0.9425535644670502, 0.902743875566844,
    #  0.9523302706523095, 0.7969517367583336, 0.6466860078687823, 0.880210793498437,
    # 0.9521142126729099, 0.9582056225658377]
    restore_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/' \
                   'liver_spleen-unet-vgg-ep30-End-loss0.1542.h5'
    base_name = os.path.basename(restore_path)
    target_name, model_name, backbone_name = base_name.split('-')[:3]
    target_name_list = target_name.split('_')
    input_layer = tf.keras.layers.Input(shape=(512, 512, 3))
    seg_model = SegmentationModel(model_name, backbone_name, 2, name=target_name, input_tensor=input_layer)
    seg_model.restore(restore_path)
    dices = []
    predictions = []
    for idx in config.EVALUATING_RANGE:
        print(idx)
        case_name = '{:04d}.nii'.format(idx)
        img_path = os.path.join(config.RAW_DATA_EVALUATING_DIR, 'img', 'img' + case_name)
        label_path = os.path.join(config.RAW_DATA_TRAINING_DIR, 'label',
                                  'label' + case_name)
        img = read_nii(img_path)
        label = read_nii(label_path)
        label_matrixs = []
        all_label_matrix = np.zeros_like(label)
        for organ_name in target_name_list:
            all_label_matrix[label == config.name2global_label[organ_name]] = 1
            zero_matrixs = np.zeros_like(label)
            zero_matrixs[label == config.name2global_label[organ_name]] = 1
            label_matrixs.append(zero_matrixs)

        window_center = config.get_dataset_config('V1')['window_center']
        window_width = config.get_dataset_config('V1')['window_width']
        window_left = window_center - window_width / 2
        window_right = window_center + window_width / 2
        img[img < window_left] = window_left
        img[img > window_right] = window_right

        img = np.transpose(img, axes=[2, 0, 1])
        label = np.transpose(all_label_matrix, axes=[2, 0, 1])
        img = np.expand_dims(img, axis=3)
        img = np.concatenate([img, img, img], axis=-1)
        img = np.asarray(img, np.float)

        prediction_probs = seg_model.predict(img, batch_size=8)
        prediction = np.argmax(prediction_probs, axis=-1)
        save_nii_path = os.path.join(config.RAW_DATA_EVALUATING_DIR, 'pred',
                                     'pred_' + target_name + '_' + case_name[:-4] + '.mhd')
        save_mhd_image(np.transpose(prediction, axes=[0, 2, 1]), save_nii_path)
        print(np.shape(img), np.shape(label), np.shape(prediction))
        print(np.max(label), np.min(label))
        print(np.max(prediction), np.min(prediction))
        dice = Utils.binary_dice(label, prediction)

        print('{}: {}'.format(case_name, dice))
        dices.append(dice)
        predictions.append(prediction)
    print('global average dice is ', np.mean(dices))
    print('dices are ', dices)
    return predictions


if __name__ == '__main__':
    evaluate_single_teacher()
    # evaluate_student()
    # evaluate_dual_teacher()