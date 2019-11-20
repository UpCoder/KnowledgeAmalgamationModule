import tensorflow as tf
from tensorflow import keras
import numpy as np
img_rows, img_cols = 28, 28


def return_whole_mnist():
    '''
    返回整个mnist数据集合
    :return:
    '''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if tf.keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    elif tf.keras.backend.image_data_format() == 'channels_last':
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    else:
        print('please check the keywords')
        assert False
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_train /= 255.
    # x_test /= 255.
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))
    return x_train, y_train, x_test, y_test


def return_part_mnist(labels):
    x_train, y_train, x_test, y_test = return_whole_mnist()
    res_x_train = []
    res_y_train = []
    res_x_test = []
    res_y_test = []
    for label in labels:
        res_x_train.extend(x_train[y_train == label])
        res_y_train.extend([label] * np.sum(y_train == label))
        res_x_test.extend(x_test[y_test == label])
        res_y_test.extend([label] * np.sum(y_test == label))
    random_index = list(range(len(res_x_train)))
    np.random.seed(2019)
    np.random.shuffle(random_index)
    res_x_train = np.asarray(res_x_train, np.float)
    res_y_train = np.asarray(res_y_train)
    res_x_train = res_x_train[random_index]
    res_y_train = res_y_train[random_index]
    res_x_test = np.asarray(res_x_test, np.float)
    res_y_test = np.asarray(res_y_test)
    return res_x_train, res_y_train, res_x_test, res_y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = return_whole_mnist()
    x_train, y_train, x_test, y_test = return_part_mnist([1, 2])
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))

