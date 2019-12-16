import os
from datasets.medical_image_utils import read_nii
import tensorflow as tf
import segmentation_models as sm
import numpy as np
sm.set_framework('tf.keras')
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class SegmentationModel:
    def __init__(self, model_name, backbone_name, num_classes, activation='softmax', input_tensor=None, name=None,
                 trainable=True):
        # build the basic segmentation model
        if input_tensor is None:
            input_tensor = tf.keras.layers.Input(shape=(512, 512, 3))

        with tf.variable_scope(name) as vc:
            if backbone_name == 'vgg':
                self.preprocessing_func = sm.get_preprocessing('vgg16')
            elif backbone_name == 'resnet':
                self.preprocessing_func = sm.get_preprocessing('resnet50')
            self.preprocessing_layer = tf.keras.layers.Lambda(lambda x: self.preprocessing_func(x))
            if model_name == 'unet':
                if backbone_name == 'vgg':
                    self.seg_model = sm.Unet('vgg16', encoder_weights='imagenet', classes=num_classes,
                                             activation=activation, input_tensor=self.preprocessing_layer(input_tensor))
                    self.encoder_output_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool',
                                                 'block5_pool']
                    self.decoder_output_names = ['decoder_stage0b_relu', 'decoder_stage1b_relu', 'decoder_stage2b_relu',
                                                 'decoder_stage3b_relu', 'decoder_stage4b_relu', 'final_conv']
                    self.encoder_outputs = []
                    self.decoder_outputs = []
                    for layer_name in self.encoder_output_names:
                        self.encoder_outputs.append(self.seg_model.get_layer(layer_name).output)
                    for layer_name in self.decoder_output_names:
                        self.decoder_outputs.append(self.seg_model.get_layer(layer_name).output)

                elif backbone_name == 'resnet':
                    self.seg_model = sm.Unet('resnet50', encoder_weights='imagenet', classes=num_classes,
                                             activation=activation, input_shape=(512, 512, 3),
                                             input_tensor=self.preprocessing_layer(input_tensor))
                    self.encoder_output_names = ['relu0', 'stage1_unit3_relu3', 'stage2_unit4_relu3',
                                                 'stage3_unit6_relu3', 'stage4_unit3_relu3']
                    self.decoder_output_names = ['decoder_stage0b_relu', 'decoder_stage1b_relu', 'decoder_stage2b_relu',
                                                 'decoder_stage3b_relu', 'decoder_stage4b_relu', 'final_conv']
                    self.encoder_outputs = []
                    self.decoder_outputs = []
                    for layer_name in self.encoder_output_names:
                        self.encoder_outputs.append(self.seg_model.get_layer(layer_name).output)
                    for layer_name in self.decoder_output_names:
                        self.decoder_outputs.append(self.seg_model.get_layer(layer_name).output)

                else:
                    print('do not support ', backbone_name)
                    assert False
            elif model_name == 'pspnet':
                if backbone_name == 'vgg':
                    self.seg_model = sm.PSPNet('vgg16', encoder_weights='imagenet', classes=num_classes,
                                               activation=activation, input_tensor=self.preprocessing_layer(input_tensor))
                    self.encoder_output_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool',
                                                 'block5_pool']
                    self.decoder_output_names = ['decoder_stage0b_relu', 'decoder_stage1b_relu', 'decoder_stage2b_relu',
                                                 'decoder_stage3b_relu', 'decoder_stage4b_relu', 'final_conv']
                    self.encoder_outputs = []
                    self.decoder_outputs = []
                    for layer_name in self.encoder_output_names:
                        self.encoder_outputs.append(self.seg_model.get_layer(layer_name).output)
                    for layer_name in self.decoder_output_names:
                        self.decoder_outputs.append(self.seg_model.get_layer(layer_name).output)

                elif backbone_name == 'resnet':
                    self.seg_model = sm.PSPNet('resnet50', encoder_weights='imagenet', classes=num_classes,
                                               activation=activation, input_tensor=self.preprocessing_layer(input_tensor))
                    self.encoder_outputs = []
                    self.decoder_outputs = []
                    self.encoder_output_names = ['relu0', 'stage1_unit3_relu3', 'stage2_unit4_relu3',
                                                 'stage3_unit6_relu3', 'stage4_unit3_relu3']
                    self.decoder_output_names = ['decoder_stage0b_relu', 'decoder_stage1b_relu', 'decoder_stage2b_relu',
                                                 'decoder_stage3b_relu', 'decoder_stage4b_relu', 'final_conv']
                    for layer_name in self.encoder_output_names:
                        self.encoder_outputs.append(self.seg_model.get_layer(layer_name).output)
                    for layer_name in self.decoder_output_names:
                        self.decoder_outputs.append(self.seg_model.get_layer(layer_name).output)

                else:
                    print('do not support ', backbone_name)
                    assert False
            else:
                print('do not support ', model_name)
                assert False
        for layer in self.seg_model.layers:
            layer._name = layer._name + '_' + name
            print(layer.name, layer.output, layer.weights)
            if not trainable:
                layer.trainable = False
        if not trainable:
            self.seg_model.trainable = False
        print(self.seg_model)
        self.prediction = self.seg_model.outputs[0]
        print('prediction is ', self.seg_model.outputs)
        # if input_tensor is not None:
        #     self.prediction = self.seg_model(self.preprocessing_layer(input_tensor))
        # else:
        #     self.prediction = None

    def restore(self, restore_path):
        self.seg_model.load_weights(restore_path)

    def predict(self, images, batch_size=8):
        return self.seg_model.predict(np.asarray(images, np.float), batch_size=batch_size,
                                      verbose=1)


def preprocessing(nii_img):
    '''
    对NII进行预处理
    :param matrix: [W, H, C]
    :return:
    '''
    window_center = 40
    window_width = 250
    window_left = window_center - window_width / 2
    window_right = window_center + window_width / 2
    nii_img[nii_img < window_left] = window_left
    nii_img[nii_img > window_right] = window_right

    img = np.transpose(nii_img, axes=[2, 0, 1])
    img = np.expand_dims(img, axis=3)
    img = np.concatenate([img, img, img], axis=-1)
    img = np.asarray(img, np.float)
    return img


class Utils:
    @staticmethod
    def binary_dice(gt, pred):
        intersection = np.sum(np.logical_and(gt == 1, pred == 1))
        union = np.sum(gt == 1) + np.sum(pred == 1)
        print('intersection is ', intersection)
        print(np.sum(gt == 1), np.sum(pred == 1))
        return 2.0 * intersection / union


def inference_single_file(nii_path, mask_path=None):
    nii_img = read_nii(nii_path)
    nii_img = preprocessing(nii_img)
    input_layer = tf.keras.layers.Input(shape=(512, 512, 3))
    restore_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/ck/V1/' \
                   'liver-unet-resnet-ep99-End-loss0.0053.h5'
    base_name = os.path.basename(restore_path)
    target_name, model_name, backbone_name = base_name.split('-')[:3]
    seg_model = SegmentationModel(model_name, backbone_name, 2, name=target_name, input_tensor=input_layer)
    seg_model.restore(restore_path)
    prediction = seg_model.predict(nii_img, batch_size=8)
    print(np.shape(prediction))
    if mask_path is not None:
        # compute the dice
        gt = read_nii(mask_path)
        gt = np.transpose(gt, axes=[2, 0, 1])
        zero_matrixs = np.zeros_like(gt)
        # multi-organ, liver label is 6
        zero_matrixs[gt == 6] = 1
        gt = zero_matrixs
        print('dice: ', Utils.binary_dice(gt, np.argmax(prediction, axis=-1)))
    return prediction


if __name__ == '__main__':
    nii_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/img/img0001.nii'
    mask_path = '/media/give/HDD3/ld/Documents/datasets/Abdomen/RawData/Training/label/label0001.nii'
    inference_single_file(nii_path, mask_path)
