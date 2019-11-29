import numpy as np
from datasets.Abdomen import config
from tqdm import tqdm
from datasets.Abdomen.tfrecords_utils import Example
from datasets.Abdomen.nii2PNG import convert_nii2png_with_label
import tensorflow as tf
import os


def exist_required_labels(label_slice):
    if np.sum(label_slice) != 0:
        return True
    return False


def convert_nii2tfrecords_with_label(case_name, labels_mapping={1: 1, 6: 2}, save_dir=None):
    img_slices, label_slices = convert_nii2png_with_label(case_name, save_dir=None, labels_mapping=labels_mapping)
    if len(img_slices) == 0:
        return 0
    try:
        with tf.python_io.TFRecordWriter(os.path.join(save_dir, case_name.split('.nii')[0]+'.tfrecords')) as writer:
            for img_slice, label_slice in tqdm(zip(img_slices, label_slices)):
                img_slice = np.expand_dims(img_slice, axis=2)
                img_slice = np.concatenate([img_slice, img_slice, img_slice], axis=-1)
                example = Example()
                example.init_values(
                    ['image', 'mask'],
                    ['np_array', 'np_array'],
                    [img_slice, label_slice]
                )
                writer.write(example.get_examples().SerializeToString())
    except Exception as e:
        print(e)
    return len(img_slices)


if __name__ == '__main__':
    total_slice_num = 0
    for case_id in range(1, 41):
        print('%04d.nii' % (case_id))
        total_slice_num += convert_nii2tfrecords_with_label('%04d.nii' % case_id, save_dir=config.RAW_DATA_TF_DIR)
    print(total_slice_num)
