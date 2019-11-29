import tensorflow as tf
import segmentation_models as sm
import numpy as np
sm.set_framework('tf.keras')


class SegmentationModel:
    def __init__(self, model_name, backbone_name, num_classes, activation='softmax', input_tensor=None, name=None):
        # build the basic segmentation model
        with tf.variable_scope(name) as vc:
            if model_name == 'unet':
                if backbone_name == 'vgg':
                    self.seg_model = sm.Unet('vgg16', encoder_weights='imagenet', classes=num_classes,
                                             activation=activation)
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
                    self.preprocessing_layer = sm.get_preprocessing('vgg16')
                elif backbone_name == 'resnet':
                    self.seg_model = sm.Unet('resnet50', encoder_weights='imagenet', classes=num_classes,
                                             activation=activation, input_shape=(512, 512, 3))
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
                    self.preprocessing_layer = sm.get_preprocessing('resnet50')
                else:
                    print('do not support ', backbone_name)
                    assert False
            elif model_name == 'pspnet':
                if backbone_name == 'vgg':
                    self.seg_model = sm.PSPNet('vgg16', encoder_weights='imagenet', classes=num_classes,
                                               activation=activation)
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
                    self.preprocessing_layer = sm.get_preprocessing('vgg16')
                elif backbone_name == 'resnet':
                    self.seg_model = sm.PSPNet('resnet50', encoder_weights='imagenet', classes=num_classes,
                                               activation=activation)
                    self.seg_model = sm.Unet('resnet50', encoder_weights=None, classes=num_classes,
                                             activation=activation, input_shape=(512, 512, 3))
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
                    self.preprocessing_layer = sm.get_preprocessing('resnet50')
                else:
                    print('do not support ', backbone_name)
                    assert False
            else:
                print('do not support ', model_name)
                assert False
        for layer in self.seg_model.layers:
            layer._name = layer._name + '_' + name
            print(layer.name, layer.output, layer.weights)

        print(self.seg_model)
        if input_tensor is not None:
            self.prediction = self.seg_model(self.preprocessing_layer(input_tensor))
        else:
            self.prediction = None

    def build_loss(self, mask_tensor):
        print('start build loss')
        if self.prediction is None:
            return None
        final_ce_loss = tf.keras.losses.sparse_categorical_crossentropy(mask_tensor, self.prediction)
        print(final_ce_loss)
        return final_ce_loss

    def restore(self, restore_path):
        self.seg_model.load_weights(restore_path)

    def predict(self, images, batch_size=8):

        return self.seg_model.predict(np.asarray(self.preprocessing_layer(images), np.float), batch_size=batch_size,
                                      verbose=1)


if __name__ == '__main__':
    seg_model = SegmentationModel('unet', 'vgg', 2, name='liver')
    print(seg_model.encoder_outputs)
    print(seg_model.decoder_outputs)

