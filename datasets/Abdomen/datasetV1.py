import os
from datasets.medical_image_utils import read_nii
import numpy as np
from datasets.Abdomen.nii2PNG import get_nii2npy_func
from datasets.Abdomen import config


class DatasetV1_Liver_Slpeen:
    def __init__(self, dataset_dir, range):
        self.imgs = []
        self.labels = []
        for idx in range:
            case_name = '{:04d}.nii'.format(idx)
            img_path = os.path.join(dataset_dir, 'img', 'img' + case_name)
            label_path = os.path.join(dataset_dir, 'label',
                                      'label' + case_name)
            img = read_nii(img_path)
            label = read_nii(label_path)
            label_matrixs = []
            all_label_matrix = np.zeros_like(label)
            target_name_list = ['liver', 'spleen']
            for idx, organ_name in enumerate(target_name_list):
                all_label_matrix[label == config.name2global_label[organ_name]] = idx + 1
                zero_matrixs = np.zeros_like(label)
                zero_matrixs[label == config.name2global_label[organ_name]] = idx + 1
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
            self.imgs.append(img)
            self.labels.append(label)