import tensorflow as tf
import segmentation_models as sm
import numpy as np
import gc
sm.set_framework('tf.keras')


def dice_coef(y_true, y_pred, smooth=1):
    """
    dice = (2*|x & y|)/ (|x|+ |y|)
         =  2*sum(|a*b|)/(sum(a^2)+sum(b^2))
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


def get_OHEM_mask(loss_tensor, selected_num):
    shape = loss_tensor.get_shape().as_list()
    ce_loss_tensor_flatten = tf.reshape(loss_tensor, [shape[0] * shape[1]])
    selected_values, selected_indices = tf.nn.top_k(ce_loss_tensor_flatten, k=selected_num)
    selected_mask = tf.cast(tf.greater_equal(loss_tensor, selected_values[-1]), tf.float32)
    return selected_mask, selected_indices


def get_batch_OHEM_mask(prediction, gt, selected_num, pre_condition_mask=None):
    '''
    根据loss的大小顺序，选择最南的样本用于训练
    :param prediction: [N, W, H, 2]
    :param gt: [N, W, H]
    :param selected_num: INT
    :param pre_condition_mask: [N, W, H]
    :return:
    '''
    ce_loss_tensor = tf.keras.losses.sparse_categorical_crossentropy(y_true=gt, y_pred=prediction)
    if pre_condition_mask is not None:
        ce_loss_tensor = ce_loss_tensor * pre_condition_mask
    selected_masks, selected_indices = tf.map_fn(lambda x: get_OHEM_mask(x, selected_num), elems=ce_loss_tensor,
                                                 dtype=(tf.float32, tf.int32))
    print(selected_masks, selected_indices)
    return selected_masks, selected_indices


def triplet_loss_OHEM(prediction, gt, feature_map):
    '''
    online compute triplet loss
    由于计算量过大，我们选择OHEM的方案，选择一部分pixel出来计算triplet loss
    :param prediction: [N, W, H, 2]
    :param gt: [N, W, H]
    :param feature_map: [N, W, H, D]
    :return: [N, W, H, X]
    '''
    selected_num = 10  # 每个输入图像选择100个像素
    neg_selected_masks, neg_selected_indices = get_batch_OHEM_mask(prediction, gt, selected_num,
                                                                   pre_condition_mask=tf.cast(tf.equal(gt, 0),
                                                                                              tf.float32))
    pos_selected_masks, pos_selected_indices = get_batch_OHEM_mask(prediction, gt, selected_num,
                                                                   pre_condition_mask=tf.cast(tf.equal(gt, 1),
                                                                                              tf.float32))
    selected_indices = tf.concat([neg_selected_indices, pos_selected_indices], axis=-1)
    selected_masks = tf.cast(tf.greater_equal(neg_selected_masks + pos_selected_masks, 1.0), tf.float32)
    tf.summary.image('selected_pixel', tf.expand_dims(tf.cast(selected_masks * 200, tf.uint8), axis=3), max_outputs=3)

    def __euclidean_distance(tensor, tensors):
        '''
        计算一个tensor和多个tensor之间的相似度
        :param tensor: [D]
        :param tensors: [N, D]
        :return:
        '''
        distance = tensor - tensors
        distance = tf.reduce_sum(tf.square(distance), axis=1)
        return distance

    def __compute_triplet_loss(anchor_feature, anchor_gt, feature_map_flatten_, gt_flatten_):
        distances = __euclidean_distance(anchor_feature, feature_map_flatten_)

        print('distances are ', distances)
        num_pairs = 5
        # select hard positive pair
        # 与当前像素是同一个类别的，属于positive pair，我们优先选择距离大的
        pos_mask = tf.cast(tf.equal(gt_flatten_, anchor_gt), dtype=tf.float32)
        # tf.stop_gradient(pos_mask)
        pos_distances = distances * pos_mask
        selected_pos_distances, selected_pos_indices = tf.nn.top_k(pos_distances, k=num_pairs)
        pos_distances_mask = tf.cast(tf.greater_equal(pos_distances, selected_pos_distances[-1]), tf.float32)
        # tf.stop_gradient(pos_distances_mask)
        pos_distances_loss = tf.reduce_sum(pos_distances * pos_distances_mask) / (
                    tf.reduce_sum(pos_distances_mask) + 1e-7)

        # minimize pos_distances_loss 使得距离最小

        # select hard negative pair
        # 与当前像素不属于同一个类别，属于negative pair，我们优先选择距离小的
        neg_mask = tf.cast(tf.not_equal(gt_flatten_, anchor_gt), dtype=tf.float32)
        # tf.stop_gradient(neg_mask)
        neg_distances = distances * neg_mask
        neg_distances = tf.reduce_max(neg_distances) - neg_distances # 将大距离的变成小距离，小距离变成大距离, 选择大距离
        neg_distances = tf.where(tf.equal(neg_distances, tf.reduce_max(neg_distances)), tf.zeros_like(neg_distances),
                                 neg_distances)
        selected_neg_distances, selected_neg_indices = tf.nn.top_k(neg_distances, k=num_pairs)
        neg_distances_mask = tf.cast(tf.greater_equal(neg_distances, selected_neg_distances[-1]), tf.float32)
        # tf.stop_gradient(neg_distances_mask)
        neg_distances_loss = tf.reduce_sum(neg_distances * neg_distances_mask) / (
            tf.reduce_sum(neg_distances_mask + 1e-7))
        # maximize distances => minimize distance
        return pos_distances_loss + neg_distances_loss

    shape = feature_map.get_shape().as_list()
    feature_map_flatten = tf.reshape(feature_map, [shape[0], shape[1] * shape[2], shape[3]], name='flatten_feature_map')
    gt_flatten = tf.reshape(gt, [shape[0], shape[1] * shape[2]])
    selected_features = tf.batch_gather(feature_map_flatten, selected_indices)
    selected_gt = tf.batch_gather(gt_flatten, selected_indices)
    selected_features_flatten = tf.reshape(selected_features, [-1, shape[3]])
    selected_gt_flatten = tf.reshape(selected_gt, [-1])
    print('mark 0')
    print(selected_features_flatten, selected_gt_flatten)
    triplet_losses = tf.map_fn(
        lambda x: __compute_triplet_loss(x[0], x[1], selected_features_flatten, selected_gt_flatten),
        elems=[selected_features_flatten, selected_gt_flatten], dtype=tf.float32, name='map_fn_level2', swap_memory=True)
    return tf.reduce_mean(triplet_losses)


def triplet_loss_resize(prediction, gt, feature_map):
    '''
    online compute triplet loss
    由于计算量过大，我们选择resize的方案
    :param prediction: [N, W, H, 2]
    :param gt: [N, W, H]
    :param feature_map: [N, W, H, D]
    :return: [N, W, H, X]
    '''
    target_size = [16, 16]
    prediction = tf.image.resize_images(prediction, target_size)
    gt = tf.squeeze(
        tf.image.resize_images(tf.expand_dims(gt, axis=3), target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    feature_map = tf.image.resize_images(feature_map, target_size)
    print('start triplet loss ', prediction, gt, feature_map)

    def __euclidean_distance(tensor, tensors):
        '''
        计算一个tensor和多个tensor之间的相似度
        :param tensor: [D]
        :param tensors: [N, D]
        :return:
        '''
        distance = tensor - tensors
        distance = tf.reduce_sum(tf.square(distance), axis=1)
        return distance

    def __compute_triplet_loss(anchor_feature, anchor_gt, gt_flatten_, feature_map_flatten_):
        distances = __euclidean_distance(anchor_feature, feature_map_flatten_)

        num_pairs = 100
        # select hard positive pair
        # 与当前像素是同一个类别的，属于positive pair，我们优先选择距离大的
        pos_distances = distances * tf.cast(tf.equal(gt_flatten_, anchor_gt), dtype=tf.float32)
        selected_pos_distances, selected_indices = tf.nn.top_k(pos_distances, k=num_pairs)
        pos_distances_mask = tf.cast(tf.greater_equal(pos_distances, selected_pos_distances[-1]), tf.float32)
        pos_distances_loss = tf.reduce_sum(pos_distances * pos_distances_mask) / (
                    tf.reduce_sum(pos_distances_mask) + 1e-7)
        # minimize pos_distances_loss 使得距离最小

        # select hard negative pair
        # 与当前像素不属于同一个类别，属于negative pair，我们优先选择距离小的
        neg_distances = distances * tf.cast(tf.not_equal(gt_flatten_, anchor_gt), dtype=tf.float32)
        neg_distances = tf.reduce_max(neg_distances) - neg_distances # 将大距离的变成小距离，小距离变成大距离, 选择大距离
        neg_distances = tf.where(tf.equal(neg_distances, tf.reduce_max(neg_distances)), tf.zeros_like(neg_distances),
                                 neg_distances)
        selected_neg_distances, selected_neg_indices = tf.nn.top_k(neg_distances, k=num_pairs)
        neg_distances_mask = tf.cast(tf.greater_equal(neg_distances, selected_neg_distances[-1]), tf.float32)
        neg_distances_loss = tf.reduce_sum(neg_distances * neg_distances_mask) / (
            tf.reduce_sum(neg_distances_mask + 1e-7))
        # maximize distances => minimize distance
        return pos_distances_loss + neg_distances_loss

    shape = feature_map.get_shape().as_list()
    feature_map_flatten = tf.reshape(feature_map, [-1, shape[-1]])
    prediction_flatten = tf.reshape(prediction, [shape[0] * shape[1] * shape[2], -1])
    gt_flatten = tf.reshape(gt, [shape[0] * shape[1] * shape[2]])

    triplet_losses = tf.map_fn(lambda x: __compute_triplet_loss(x[0], x[1], gt_flatten, feature_map_flatten),
                               elems=[feature_map_flatten, gt_flatten], dtype=tf.float32)
    return tf.reduce_mean(triplet_losses)


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
        # if input_tensor is not none:
        #     self.prediction = self.seg_model(self.preprocessing_layer(input_tensor))
        # else:
        #     self.prediction = none

    def build_loss(self, mask_tensor, cross_entropy_coff=1.0, dice_coff=1.0, focal_loss_coff=1.0, triplet_loss_coff=1.0):
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

        # triplet_loss_t = tf.keras.layers.Lambda(lambda x: triplet_loss_OHEM(*x))((self.prediction, mask_tensor,
        #                                                                           self.decoder_outputs[-2]))
        # print('triplet_loss_t is ', triplet_loss_t)
        # tf.summary.scalar('loss/triplet_loss', triplet_loss_t)

        final_loss = cross_entropy_coff * tf.reduce_mean(
            ce_loss) + dice_coff * dice_loss + focal_loss_coff * tf.reduce_mean(
            focal_loss_t)

        return final_loss

    def restore(self, restore_path):
        self.seg_model.load_weights(restore_path, by_name=True)

    def predict(self, images, batch_size=8):
        return self.seg_model.predict(np.asarray(images, np.float), batch_size=batch_size,
                                      verbose=1)

    @staticmethod
    def evaluate(prediction, label, metric_func, case_name='01', metrics=None):
        dice = metric_func(label, prediction)

        print('{}: {}'.format(case_name, dice))
        if metrics is not None:
            metrics.append(dice)
            # predictions.append(prediction)


class Upsampling:

    def __init__(self, filters, output_size):
        self.filters = filters
        self.output_size = output_size
        self.resize_layer = tf.keras.layers.Lambda(lambda x: tf.image.resize_images(x, output_size))
        self.conv_layer = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=[1, 1], padding='SAME',
                                                 activation=None)
        self.bn_layer = tf.keras.layers.BatchNormalization()
        self.activation_layer = tf.keras.layers.ReLU()

    def call(self, input_tensor):
        x = self.resize_layer(input_tensor)
        x = self.conv_layer(x)
        # x = self.bn_layer(x)
        x = self.activation_layer(x)
        return x


class Decoder:

    def __init__(self, input_tensors=None, output_dim=3, num_filter=256):
        self.input_tensors = input_tensors[::-1]
        # self.filters = [input_tensor.get_shape().as_list()[-1] for input_tensor in self.input_tensors]
        self.output_sizes = [[i * 2 for i in input_tensor.get_shape().as_list()[1:3]] for input_tensor in
                             self.input_tensors]
        print(self.output_sizes)
        # self.upsamplings = [Upsampling(filter, output_size) for filter, output_size in
        #                     zip(self.filters, self.output_sizes)]
        self.upsamplings = [Upsampling(num_filter, output_size) for output_size in self.output_sizes]
        self.final_conv = tf.keras.layers.Conv2D(filters=output_dim, kernel_size=[3, 3], padding='SAME')

    def call(self):
        last_tensor = None
        for input_tensor, upsampling_layer in zip(self.input_tensors, self.upsamplings):
            if last_tensor is not None:
                cur_tensor = tf.keras.layers.Concatenate()([last_tensor, input_tensor])
            else:
                cur_tensor = input_tensor
            cur_tensor = upsampling_layer.call(cur_tensor)
            last_tensor = cur_tensor
            print('last tensor is ', last_tensor)
        return self.final_conv(last_tensor)


class SegmentationClsModel:

    '''
    包括两个decoder，一个区分前景背景，一个区分具体的类别
    '''
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
            self.cls_logits = Decoder(self.encoder_outputs).call()
            self.cls_prediction = tf.keras.layers.Softmax()(self.cls_logits)
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
        # if input_tensor is not none:
        #     self.prediction = self.seg_model(self.preprocessing_layer(input_tensor))
        # else:
        #     self.prediction = none

    def build_loss(self, mask_tensor, gen_tensor, cross_entropy_coff=1.0, dice_coff=1.0, focal_loss_coff=1.0):
        print('start build loss')
        if self.prediction is None:
            return None
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(mask_tensor, self.prediction)
        tf.summary.scalar('loss/sg/final_ce', tf.reduce_mean(ce_loss))

        dice_loss = dice_coef_loss(tf.cast(mask_tensor, tf.float32), self.prediction[:, :, :, 1])
        print('dice_loss is ', dice_loss)
        tf.summary.scalar('loss/sg/final_dice_loss', dice_loss)

        focal_loss_t = focal_loss()(tf.cast(mask_tensor, tf.float32), self.prediction[:, :, :, 1])
        tf.summary.scalar('loss/sg/focal_loss', tf.reduce_mean(focal_loss_t))
        final_sg_loss = cross_entropy_coff * tf.reduce_mean(
            ce_loss) + dice_coff * dice_loss + focal_loss_coff * tf.reduce_mean(focal_loss_t)

        cls_ce_loss = tf.keras.losses.sparse_categorical_crossentropy(gen_tensor, self.cls_prediction)
        tf.summary.scalar('loss/cls/final_ce', tf.reduce_mean(cls_ce_loss))

        final_loss = final_sg_loss + tf.reduce_mean(cls_ce_loss)
        return final_loss

    def restore(self, restore_path):
        self.seg_model.load_weights(restore_path)

    def predict(self, images, batch_size=8):
        custom_model = tf.keras.models.Model(self.seg_model.inputs, [self.prediction, self.cls_prediction])
        return custom_model.predict(np.asarray(images, np.float), batch_size=batch_size,
                                    verbose=1)

    @staticmethod
    def evaluate(prediction, label, metric_func, case_name='01', metrics=None):
        dice = metric_func(label, prediction)

        print('{}: {}'.format(case_name, dice))
        if metrics is not None:
            metrics.append(dice)
            # predictions.append(prediction)


if __name__ == '__main__':
    seg_model = SegmentationModel('unet', 'vgg', 2, name='liver')
    print(seg_model.encoder_outputs)
    print(seg_model.decoder_outputs)

