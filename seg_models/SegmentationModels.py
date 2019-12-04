import tensorflow as tf
import segmentation_models as sm
import numpy as np
sm.set_framework('tf.keras')


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (
                tf.keras.backend.sum(tf.keras.backend.square(y_true)) + tf.keras.backend.sum(
            tf.keras.backend.square(y_pred)) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.keras.backend.mean(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(
            tf.keras.backend.epsilon() + pt_1)) - tf.keras.backend.mean(
            (1 - alpha) * tf.keras.backend.pow(pt_0, gamma) * tf.keras.backend.log(
                1. - pt_0 + tf.keras.backend.epsilon()))
    return focal_loss_fixed


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

    def build_loss(self, mask_tensor, cross_entropy_coff=1.0, dice_coff=1.0, focal_loss_coff=1.0):
        print('start build loss')
        if self.prediction is None:
            return None
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(mask_tensor, self.prediction)
        tf.summary.scalar('loss/final_ce', tf.reduce_mean(ce_loss))

        dice_loss = dice_coef_loss(tf.cast(mask_tensor, tf.float32), self.prediction[:, :, :, 1])
        print('dice_loss is ', dice_loss)
        tf.summary.scalar('loss/final_dice_loss', dice_loss)

        focal_loss_t = focal_loss()(tf.cast(mask_tensor, tf.float32), self.prediction[:, :, :, 1])
        tf.summary.scalar('loss/focal_loss', tf.reduce_mean(focal_loss_t))
        final_loss = cross_entropy_coff * tf.reduce_mean(
            ce_loss) + dice_coff * dice_loss + focal_loss_coff * tf.reduce_mean(focal_loss_t)
        return final_loss

    def restore(self, restore_path):
        self.seg_model.load_weights(restore_path)

    def predict(self, images, batch_size=8):
        return self.seg_model.predict(np.asarray(images, np.float), batch_size=batch_size,
                                      verbose=1)


if __name__ == '__main__':
    seg_model = SegmentationModel('unet', 'vgg', 2, name='liver')
    print(seg_model.encoder_outputs)
    print(seg_model.decoder_outputs)

