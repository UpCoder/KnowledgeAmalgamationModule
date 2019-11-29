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


def convert_nii2png_with_label(case_name, labels_mapping={1: 1, 6: 2}, save_dir=None):
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
    window_center = config.get_dataset_config('V1')['window_center']
    window_width = config.get_dataset_config('V1')['window_width']
    window_left = window_center - window_width / 2
    window_right = window_center + window_width / 2
    img[img < window_left] = window_left
    img[img > window_right] = window_right

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


if __name__ == '__main__':
    img_slices, label_slices = convert_nii2png_with_label('0001.nii', save_dir=config.RAW_DATA_TRAINING_DIR)
    print(np.shape(img_slices), np.shape(label_slices))