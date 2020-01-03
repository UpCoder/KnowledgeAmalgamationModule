import numpy as np
from datasets.medical_image_utils import read_nii
from datasets.Abdomen import config
import os
import cv2
from tqdm import tqdm


def exist_required_labels(label_slice):
    if np.sum(label_slice) != 0:
        return True
    return False


def get_nii2npy_func(name):
    if name == 'V1' or name == 'v1':
        return Nii2Npys.convert_nii2png_with_label_v1
    if name == 'V2' or name == 'v2':
        return Nii2Npys.convert_nii2png_with_label_v2
    if name == 'V3' or name == 'v3':
        return Nii2Npys.convert_nii2png_with_label_v3
    print('name is incorrect, ', name)
    return None


class Nii2Npys:
    @staticmethod
    def convert_nii2png_with_label_v2(case_name, labels_mapping={1: 1, 6: 2}, save_dir=None, is_training=True,
                                      img_prefix='img', label_prefix='label', Training_DIR=None):
        '''
        1、保存有organ 的slice和 与organ slice数量一半的无器官的slice
        2、organ 结尾的slice和开始的slice 重复取总个数的10%
        :param case_name: case name, such as '0001.nii'
        :param labels_mapping: label的映射关系
        :param save_dir: 保存的目录
        :param is_training
        :return:
        '''

        img_path = os.path.join(Training_DIR, 'img', img_prefix + case_name)
        label_path = os.path.join(Training_DIR, 'label',
                                  label_prefix + case_name)
        if not os.path.exists(img_path):
            print(img_path, ' does not exists')
            return None, None
        img = read_nii(img_path)
        label = read_nii(label_path)
        print(np.shape(img), np.shape(label))
        # 窗宽窗位
        window_center = config.get_dataset_config('V2')['window_center']
        window_width = config.get_dataset_config('V2')['window_width']
        window_left = window_center - window_width / 2
        window_right = window_center + window_width / 2
        img[img < window_left] = window_left
        img[img > window_right] = window_right
        if not is_training:
            imgs = []
            for idx in range(np.shape(img)[-1]):
                imgs.append(
                    np.concatenate(
                        [np.expand_dims(img[:, :, idx], axis=2),
                         np.expand_dims(img[:, :, idx], axis=2),
                         np.expand_dims(img[:, :, idx], axis=2)], axis=2
                    )
                )
            label = np.transpose(label, axes=[2, 0, 1])
            return np.asarray(imgs, np.float), np.asarray(label)

        # label mapping
        zero_matrix = np.zeros_like(label)
        for key in labels_mapping.keys():
            zero_matrix[label == key] = labels_mapping[key]
        label = zero_matrix

        # 找到所有的包含器官的slice
        organ_imgs = []
        organ_labels = []
        organ_idxs = []
        for idx in tqdm(range(1, np.shape(img)[-1]-1)):
            label_slice = label[:, :, idx]
            img_slice = img[:, :, idx]
            if exist_required_labels(label_slice):
                organ_imgs.append(img_slice)
                organ_labels.append(label_slice)
                organ_idxs.append(idx)
        print('{} slices contain organ'.format(len(organ_imgs)))

        # 找到所有不包含器官的slice
        without_organ_imgs = []
        without_organ_labels = []
        all_idxs = list(range(np.shape(img)[-1]))
        without_organ_idxs = list(set(all_idxs) - set(organ_idxs))
        np.random.shuffle(without_organ_idxs)
        # for idx in without_organ_idxs[:int(len(organ_idxs)*0.5)]:
        for idx in without_organ_idxs:
            label_slice = label[:, :, idx]
            img_slice = img[:, :, idx]
            without_organ_labels.append(label_slice)
            without_organ_imgs.append(img_slice)
        print('{} slices without organ'.format(len(without_organ_imgs)))
        # 找到每个器官开始的slice
        repeat_organ_imgs = []
        repeat_organ_labels = []
        for value in labels_mapping.values():
            for img_slice, label_slice in zip(organ_imgs, organ_labels):
                if value in label_slice:
                    repeat_organ_imgs.extend([img_slice] * int(len(organ_imgs) * 0.1))
                    repeat_organ_labels.extend([label_slice] * int(len(organ_imgs) * 0.1))
                    break
            for img_slice, label_slice in zip(organ_imgs[::-1], organ_labels[::-1]):
                if value in label_slice:
                    repeat_organ_imgs.extend([img_slice] * int(len(organ_imgs) * 0.1))
                    repeat_organ_labels.extend([label_slice] * int(len(organ_imgs) * 0.1))
                    break
        print('{} slices first/last organ slice'.format(len(repeat_organ_imgs)))
        return_imgs = []
        return_labels = []
        return_imgs.extend(organ_imgs)
        return_imgs.extend(without_organ_imgs)
        return_imgs.extend(repeat_organ_imgs)

        return_labels.extend(organ_labels)
        return_labels.extend(without_organ_labels)
        return_labels.extend(repeat_organ_labels)
        if save_dir is not None:
            for idx, (img, label) in enumerate(zip(return_imgs, return_labels)):
                img_png_path = os.path.join(save_dir, 'PNGs/img',
                                            case_name + '_' + str(idx) + '.png')
                label_png_path = os.path.join(save_dir, 'PNGs/label',
                                              case_name + '_' + str(idx) + '.png')
                label_vis_png_path = os.path.join(save_dir, 'PNGs/label_vis',
                                                  case_name + '_' + str(idx) + '.png')
                # print(img_png_path)
                cv2.imwrite(img_png_path, np.asarray(img, np.int))
                cv2.imwrite(label_png_path, np.asarray(label, np.int))
                cv2.imwrite(label_vis_png_path, np.asarray(label * 100, np.int))
        return np.asarray(return_imgs, np.int), np.asarray(return_labels, np.int)

    @staticmethod
    def convert_nii2png_with_label_v1(case_name, labels_mapping={1: 1, 6: 2}, save_dir=None, is_training=True,
                                      img_prefix='img', label_prefix='label', Training_DIR=None):
        '''
        1、只保存有organ 的slice
        2、每个example 只取一层
        :param case_name: case name, such as '0001.nii'
        :param labels_mapping: label的映射关系
        :param img_prefix:
        :param label_prefix
        :param Training_DIR
        :param save_dir: 保存的目录
        :return:
        '''
        img_path = os.path.join(Training_DIR, 'img', img_prefix + case_name)
        label_path = os.path.join(Training_DIR, 'label',
                                  label_prefix + case_name)
        if not os.path.exists(img_path):
            print(img_path, ' does not exists')
            return [], []
        img = read_nii(img_path)
        label = read_nii(label_path)
        print(np.shape(img), np.shape(label))
        # 窗宽窗位
        window_center = config.get_dataset_config('V1')['window_center']
        window_width = config.get_dataset_config('V1')['window_width']
        window_left = window_center - window_width / 2
        window_right = window_center + window_width / 2
        img[img < window_left] = window_left
        img[img > window_right] = window_right
        if not is_training:
            img = np.transpose(img, axes=[2, 0, 1])
            label = np.transpose(label, axes=[2, 0, 1])
            img = np.expand_dims(img, axis=3)
            img = np.concatenate([img, img, img], axis=-1)
            img = np.asarray(img, np.float)
            return img, label
        # label mapping
        zero_matrix = np.zeros_like(label)
        for key in labels_mapping.keys():
            zero_matrix[label == key] = labels_mapping[key]
        label = zero_matrix
        return_imgs = []
        return_labels = []
        for idx in tqdm(range(1, np.shape(img)[-1]-1)):
            label_slice = label[:, :, idx]
            img_slice = img[:, :, idx]
            if exist_required_labels(label_slice):
                if save_dir is not None:
                    img_png_path = os.path.join(save_dir, 'PNGs/img',
                                                case_name + '_' + str(idx) + '.png')
                    label_png_path = os.path.join(save_dir, 'PNGs/label',
                                                  case_name + '_' + str(idx) + '.png')
                    label_vis_png_path = os.path.join(save_dir, 'PNGs/label_vis',
                                                      case_name + '_' + str(idx) + '.png')
                    # print(img_png_path)
                    cv2.imwrite(img_png_path, np.asarray(img_slice, np.int))
                    cv2.imwrite(label_png_path, np.asarray(label_slice, np.int))
                    cv2.imwrite(label_vis_png_path, np.asarray(label_slice * 100, np.int))
                return_imgs.append(img_slice)
                return_labels.append(label_slice)
        return np.asarray(return_imgs, np.int), np.asarray(return_labels, np.int)

    @staticmethod
    def convert_nii2png_with_label_v3(case_name, labels_mapping={1: 1, 6: 2}, save_dir=None, is_training=True):
        '''
        1、保存有organ 的slice和 与organ slice数量一半的无器官的slice
        2、organ 结尾的slice和开始的slice 重复取总个数的10%
        3、无窗宽 窗位
        :param case_name: case name, such as '0001.nii'
        :param labels_mapping: label的映射关系
        :param save_dir: 保存的目录
        :param is_training
        :return:
        '''
        img_path = os.path.join(config.RAW_DATA_TRAINING_DIR, 'img', 'img' + case_name)
        label_path = os.path.join(config.RAW_DATA_TRAINING_DIR, 'label',
                                  'label' + case_name)
        if not os.path.exists(img_path):
            print(img_path, ' does not exists')
            return [], []
        img = read_nii(img_path)
        label = read_nii(label_path)
        print(np.shape(img), np.shape(label))
        # 窗宽窗位
        window_center = config.get_dataset_config('V3')['window_center']
        window_width = config.get_dataset_config('V3')['window_width']
        if window_center is not None:
            window_left = window_center - window_width / 2
            window_right = window_center + window_width / 2
            img[img < window_left] = window_left
            img[img > window_right] = window_right
        if not is_training:
            imgs = []
            for idx in range(np.shape(img)[-1]):
                imgs.append(
                    np.concatenate(
                        [np.expand_dims(img[:, :, idx], axis=2),
                         np.expand_dims(img[:, :, idx], axis=2),
                         np.expand_dims(img[:, :, idx], axis=2)], axis=2
                    )
                )
            label = np.transpose(label, axes=[2, 0, 1])
            return np.asarray(imgs, np.float), np.asarray(label)

        # label mapping
        zero_matrix = np.zeros_like(label)
        for key in labels_mapping.keys():
            zero_matrix[label == key] = labels_mapping[key]
        label = zero_matrix

        # 找到所有的包含器官的slice
        organ_imgs = []
        organ_labels = []
        organ_idxs = []
        for idx in tqdm(range(1, np.shape(img)[-1] - 1)):
            label_slice = label[:, :, idx]
            img_slice = img[:, :, idx]
            if exist_required_labels(label_slice):
                organ_imgs.append(img_slice)
                organ_labels.append(label_slice)
                organ_idxs.append(idx)
        print('{} slices contain organ'.format(len(organ_imgs)))

        # 找到所有不包含器官的slice
        without_organ_imgs = []
        without_organ_labels = []
        all_idxs = list(range(np.shape(img)[-1]))
        without_organ_idxs = list(set(all_idxs) - set(organ_idxs))
        np.random.shuffle(without_organ_idxs)
        for idx in without_organ_idxs[:int(len(organ_idxs) * 0.5)]:
            label_slice = label[:, :, idx]
            img_slice = img[:, :, idx]
            without_organ_labels.append(label_slice)
            without_organ_imgs.append(img_slice)
        print('{} slices without organ'.format(len(without_organ_imgs)))
        # 找到每个器官开始的slice
        repeat_organ_imgs = []
        repeat_organ_labels = []
        for value in labels_mapping.values():
            for img_slice, label_slice in zip(organ_imgs, organ_labels):
                if value in label_slice:
                    repeat_organ_imgs.extend([img_slice] * int(len(organ_imgs) * 0.1))
                    repeat_organ_labels.extend([label_slice] * int(len(organ_imgs) * 0.1))
                    break
            for img_slice, label_slice in zip(organ_imgs[::-1], organ_labels[::-1]):
                if value in label_slice:
                    repeat_organ_imgs.extend([img_slice] * int(len(organ_imgs) * 0.1))
                    repeat_organ_labels.extend([label_slice] * int(len(organ_imgs) * 0.1))
                    break
        print('{} slices first/last organ slice'.format(len(repeat_organ_imgs)))
        return_imgs = []
        return_labels = []
        return_imgs.extend(organ_imgs)
        return_imgs.extend(without_organ_imgs)
        return_imgs.extend(repeat_organ_imgs)

        return_labels.extend(organ_labels)
        return_labels.extend(without_organ_labels)
        return_labels.extend(repeat_organ_labels)
        if save_dir is not None:
            for idx, (img, label) in enumerate(zip(return_imgs, return_labels)):
                img_png_path = os.path.join(save_dir, 'PNGs/img',
                                            case_name + '_' + str(idx) + '.png')
                label_png_path = os.path.join(save_dir, 'PNGs/label',
                                              case_name + '_' + str(idx) + '.png')
                label_vis_png_path = os.path.join(save_dir, 'PNGs/label_vis',
                                                  case_name + '_' + str(idx) + '.png')
                # print(img_png_path)
                cv2.imwrite(img_png_path, np.asarray(img, np.int))
                cv2.imwrite(label_png_path, np.asarray(label, np.int))
                cv2.imwrite(label_vis_png_path, np.asarray(label * 100, np.int))
        return np.asarray(return_imgs, np.int), np.asarray(return_labels, np.int)


if __name__ == '__main__':
    img_slices, label_slices = Nii2Npys.convert_nii2png_with_label_v3('0022.nii', save_dir=config.RAW_DATA_TRAINING_DIR)
    print(np.shape(img_slices), np.shape(label_slices))