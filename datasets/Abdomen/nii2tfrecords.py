import numpy as np
from datasets.Abdomen import config
from tqdm import tqdm
from datasets.Abdomen.tfrecords_utils import Example
from datasets.Abdomen.nii2PNG import get_nii2npy_func
import tensorflow as tf
import os
import cv2


def exist_required_labels(label_slice):
    if np.sum(label_slice) != 0:
        return True
    return False


def convert_nii2tfrecords_with_label(case_name, dataset_name=None, name='V1'):
    if dataset_name is None:
        return
    dataset_config = config.getDatasetConfigFactory(dataset_name)
    img_slices, label_slices = get_nii2npy_func(name)(case_name, save_dir=None,
                                                      labels_mapping=dataset_config.labels_mapping,
                                                      img_prefix=dataset_config.img_prefix,
                                                      Training_DIR=dataset_config.RAW_DATA_TRAINING_DIR,
                                                      label_prefix=dataset_config.label_prefix)
    if img_slices is None:
        return 0
    try:
        with tf.python_io.TFRecordWriter(os.path.join(dataset_config.RAW_DATA_TF_DIR, name,
                                                      case_name.split('.nii')[0] + '.tfrecords')) as writer:
            for img_slice, label_slice in tqdm(zip(img_slices, label_slices)):
                if name == 'V1' or name == 'V2' or name == 'V3':
                    img_slice = np.expand_dims(img_slice, axis=2)
                    img_slice = np.concatenate([img_slice, img_slice, img_slice], axis=-1)
                w, h, c = np.shape(img_slice)
                img_slice = np.asarray(img_slice, np.float)
                label_slice = np.asarray(label_slice, np.int)
                if w != 512 or h != 512:
                    img_slice = cv2.resize(img_slice, (512, 512))
                    label_slice = cv2.resize(label_slice, (512, 512), interpolation=cv2.INTER_NEAREST)
                img_slice = np.asarray(img_slice, np.int)
                label_slice = np.asarray(label_slice, np.int)
                example = Example()
                example.init_values(
                    ['image', 'mask'],
                    ['np_array', 'np_array'],
                    [img_slice, label_slice]
                )
                writer.write(example.get_examples().SerializeToString())
    except Exception as e:
        print(e)
    print(np.shape(img_slices))
    return len(img_slices)


if __name__ == '__main__':
    total_slice_num = 0
    for case_id in range(20, 40):
        print('%04d.nii' % (case_id))
        total_slice_num += convert_nii2tfrecords_with_label(
            '%04d.nii' % case_id,
            dataset_name='Abdomen',
            name='V2')
    print(total_slice_num)
