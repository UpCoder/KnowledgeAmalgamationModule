import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import os
from glob import glob
from datasets.Abdomen import config
from datasets.tf_augmentation import get_aug_tensor


def read_tfrecords(tf_record_paths):
    '''
    :param tf_record_path:
    :return:
    '''
    keys_to_features = {
        'image': tf.VarLenFeature(tf.int64),
        'mask': tf.VarLenFeature(tf.int64),
        # 'width': tf.FixedLenFeature((), tf.int64),
        # 'channel': tf.FixedLenFeature((), tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Tensor('image'),
        'mask': slim.tfexample_decoder.Tensor('mask')

    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    reader = tf.TFRecordReader
    items_to_descriptions = {
        'image': 'A color image of varying height and width.',
        'mask': 'Shape of the image',
    }

    return slim.dataset.Dataset(
        data_sources=tf_record_paths,
        reader=reader,
        decoder=decoder,
        num_samples=1,
        items_to_descriptions=items_to_descriptions,
        )


def _parse_function(proto, version_name):

    # define your tfrecord again. Remember that you saved your image as a string.
    # keys_to_features = {'image': tf.FixedLenFeature([], tf.string),
    #                     "label": tf.FixedLenFeature([], tf.int64)}
    #
    keys_to_features = {
        # 'image': tf.VarLenFeature(tf.int64),
        # 'mask': tf.VarLenFeature(tf.int64),
        'image': tf.FixedLenFeature([512, 512, 3], tf.int64),
        'mask': tf.FixedLenFeature([512, 512], tf.int64)
    }
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    image = parsed_features['image']
    mask = parsed_features['mask']
    print('image is ', image)
    print('mask is ', mask)
    # todo augmentation
    if config.get_dataset_config(version_name)['prob'] is not None:
        mask = tf.expand_dims(mask, axis=2)
        image, mask = get_aug_tensor(image, mask, prob=config.get_dataset_config(version_name)['prob'])
        mask = tf.squeeze(mask, axis=2)
    return image, mask


def parse_tfrecords(dataset_dir, shuffle_size, batch_size, prefetch_size):
    tfrecord_paths = []
    tfrecord_paths.extend(glob(os.path.join(dataset_dir, '*.tfrecords')))
    version_name = os.path.basename(dataset_dir)
    tfrecord_paths = list(set(tfrecord_paths))
    print('the tfrecord_paths are ', tfrecord_paths)
    print('the len tfrecord_paths is ', len(tfrecord_paths))
    np.random.seed(2019)
    np.random.shuffle(tfrecord_paths)
    dataset = tf.data.TFRecordDataset(tfrecord_paths)

    dataset = dataset.map(lambda x: _parse_function(x, version_name), num_parallel_calls=16)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(shuffle_size, seed=2019)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    iterator = dataset.make_one_shot_iterator()
    images, masks = iterator.get_next()

    images = tf.reshape(images, [batch_size, 512, 512, 3])
    masks = tf.reshape(masks, [batch_size, 512, 512])

    images = tf.cast(images, tf.float32)
    masks = tf.cast(masks, tf.uint8)
    print('batch_images are ', images)
    print('batch_masks are ', masks)
    return images, masks


def main_reader():
    from glob import glob
    dataset_dir = [
        '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/tfrecords',
    ]

    tfrecord_paths = []
    for dataset_dir in dataset_dir:
        tfrecord_paths.extend(glob(os.path.join(dataset_dir, '0001.tfrecords')))
    tfrecord_paths = list(set(tfrecord_paths))
    # tfrecord_paths.sort(key=lambda f: int(filter(str.isdigit, f)))
    print('the tfrecord_paths are ', tfrecord_paths)
    print('the len tfrecord_paths is ', len(tfrecord_paths))
    # tfrecord_paths = tfrecord_paths[45:50]
    np.random.seed(2019)
    np.random.shuffle(tfrecord_paths)

    dataset = read_tfrecords(tf_record_paths=tfrecord_paths)
    batch_size = 32
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, num_readers=1, common_queue_capacity=batch_size * 1000, common_queue_min=batch_size * 100, shuffle=True)

    images, masks = provider.get(['image', 'mask'])
    print(images, masks)
    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, [512, 512, 3])
    masks = tf.reshape(masks, [512, 512])
    # i_array.set_shape([None, None, 3])

    b_image, b_mask = tf.train.batch([images, masks], batch_size=batch_size,
                              num_threads=1, capacity=64)

    batch_queue = slim.prefetch_queue.prefetch_queue([b_image, b_mask],
                                                     capacity=64)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    thread = tf.train.start_queue_runners(sess=sess)

    iarray = batch_queue.dequeue()
    for i in range(10000):
        images, masks = sess.run(iarray)
        print(i, np.shape(images), np.shape(masks))


if __name__ == '__main__':
    main_reader()