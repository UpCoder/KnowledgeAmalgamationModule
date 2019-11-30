import tensorflow as tf
import cv2
import numpy as np


class FeatureAlignmentModule:
    def __init__(self, num_filter, require_resize=False):
        self.conv_layer_1 = tf.keras.layers.Conv2D(num_filter, kernel_size=(1, 1), strides=[1, 1], activation=None,
                                                   trainable=True, use_bias=False)
        self.conv_layer_2 = tf.keras.layers.Conv2D(num_filter, kernel_size=(1, 1), strides=[1, 1], activation=None,
                                                   trainable=True, use_bias=False)
        self.require_resize = require_resize
        if self.require_resize:
            self.resize_layer = tf.keras.layers.Lambda(lambda x: tf.image.resize_images(x, [512, 512]))

    def call(self, input_tensor1, input_tensor2):
        output_1 = self.conv_layer_1(input_tensor1)
        if self.require_resize:
            output_1 = self.resize_layer(output_1)
        output_2 = self.conv_layer_2(input_tensor2)
        if self.require_resize:
            output_2 = self.resize_layer(output_2)
        return output_1, output_2, self.conv_layer_1.weights[0], self.conv_layer_2.weights[0]

    @staticmethod
    def block_loss(t_o, s_o, t_weight, s_weight, pixel_wise=False):
        if pixel_wise:
            loss_1 = tf.keras.losses.mean_squared_error(t_o, s_o)
        else:
            loss_1 = tf.keras.backend.mean(tf.keras.losses.mean_squared_error(t_o, s_o), axis=[1, 2])
        loss_reg_t = tf.keras.backend.mean(tf.keras.losses.mean_squared_error(t_weight, 1.))
        loss_reg_s = tf.keras.backend.mean(tf.keras.losses.mean_squared_error(s_weight, 1.))
        print(loss_1, loss_reg_t, loss_reg_s)
        return loss_1 + loss_reg_t + loss_reg_s


def put_text(imgs, teacher1_pred, teacher2_pred, student_pred, teacher1_entropy, teacher2_entropy):
    result = np.zeros([imgs.shape[0], 512, 512], dtype=np.int)
    # print(np.shape(imgs))
    for i in range(imgs.shape[0]):
        show_string = 'teacher1 out: %d\nteacher2 out: %d\nstudent out: %d\nteacher1 entropy: %.4f\nteacher2 entropy: %.4f\n' % (
        teacher1_pred[i], teacher2_pred[i], student_pred[i], teacher1_entropy[i], teacher2_entropy[i])
        img = result[i, :, :] * 255
        y0, dy = 50, 25
        res_mat = img
        for j, txt in enumerate(show_string.split('\n')):
            y = y0 + j * dy
            res_mat = cv2.putText(res_mat, str(txt), (50, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
        result[i, :, :] = res_mat.get()
    return np.asarray(result, np.float)


def tf_put_text(imgs, teacher1_pred, teacher2_pred, student_pred, teacher1_entropy, teacher2_entropy):
    return tf.py_func(put_text, [imgs, teacher1_pred, teacher2_pred, student_pred, teacher1_entropy, teacher2_entropy],
                      Tout=imgs.dtype)


if __name__ == '__main__':
    from dataset import return_part_mnist
    res_x_train, res_y_train, res_x_test, res_y_test = return_part_mnist([0, 1])
    results = put_text(res_x_train[:2], [0, 1], [0, 1], [0, 1], [0.00, 0.23], [0.00, 0.22])
    print(np.shape(results))
    for idx, result in enumerate(results):
        cv2.imwrite('./%d.jpg' % idx, result)