import tensorflow as tf


def triplet_loss(prediction, gt, feature_map):
    '''
    online compute triplet loss
    :param prediction: [N, W, H, 2]
    :param gt: [N, W, H]
    :param feature_map: [N, W, H, D]
    :return: [N, W, H, X]
    '''
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
        print(pos_distances_loss, neg_distances_loss)
        return pos_distances_loss + neg_distances_loss

    shape = feature_map.get_shape().as_list()
    feature_map_flatten = tf.reshape(feature_map, [-1, shape[-1]])
    prediction_flatten = tf.reshape(prediction, [shape[0] * shape[1] * shape[2], -1])
    gt_flatten = tf.reshape(gt, [shape[0] * shape[1] * shape[2]])

    triplet_losses = tf.map_fn(lambda x: __compute_triplet_loss(x[0], x[1], gt_flatten, feature_map_flatten),
                               elems=[feature_map_flatten, gt_flatten], dtype=tf.float32)
    return tf.reduce_mean(triplet_losses)


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
    selected_num = 100  # 每个输入图像选择100个像素
    selected_masks, selected_indices = get_batch_OHEM_mask(prediction, gt, selected_num)

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

    def __single_triplet_loss(single_feature_flatten, single_gt_flatten, single_selected_indices):
        '''
        :param single_feature_flatten: [W*H, D]
        :param single_gt_flatten: [W*H]
        :param selected_indices: [100]
        :return:
        '''
        single_selected_features = tf.gather(single_feature_flatten, single_selected_indices)
        single_selected_gt = tf.gather(single_gt_flatten, single_selected_indices)
        single_triplet_loss = tf.map_fn(
            lambda x: __compute_triplet_loss(x[0], x[1], single_selected_gt, single_selected_features),
            elems=[single_selected_features, single_selected_gt], dtype=tf.float32)
        return tf.reduce_mean(single_triplet_loss)

    shape = feature_map.get_shape().as_list()
    feature_map_flatten = tf.reshape(feature_map, [shape[0], shape[1] * shape[2], shape[-1]])
    gt_flatten = tf.reshape(gt, [shape[0], shape[1] * shape[2]])
    triplet_losses = tf.map_fn(lambda x: __single_triplet_loss(x[0], x[1], x[2]),
                               elems=[feature_map_flatten, gt_flatten, selected_indices], dtype=tf.float32)
    return tf.reduce_mean(triplet_losses)


def success():
    import tensorflow as tf
    import numpy as np

    def test_func(i):
        return tf.cast(i, tf.int32), tf.add(tf.cast(i, tf.float32), 1.2)

    test_range = tf.constant(np.arange(5))

    with tf.device("/cpu:0"):
        test = tf.map_fn(test_func, test_range, dtype=(tf.int32, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(test)


def failed():
    import tensorflow as tf
    import numpy as np

    def test_func(i):
        return tf.cast(i, tf.int32), tf.add(tf.cast(i, tf.float32), 1.2)

    test_range = tf.constant(np.arange(5))

    with tf.device("/gpu:0"):
        test = tf.map_fn(test_func, test_range, dtype=(tf.int32, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(test)


if __name__ == '__main__':
    # import tensorflow.contrib.eager as tfe
    #
    # tfe.enable_eager_execution()
    # prediction_tensor = tf.zeros([1, 512, 512, 2], tf.float32)
    # gt_tensor = tf.zeros([1, 512, 512], tf.int32)
    # feature_map_tensor = tf.zeros([1, 512, 512, 256], tf.float32)
    # # triplet_loss(prediction_tensor, gt_tensor, feature_map_tensor)
    # # get_batch_OHEM_mask(prediction_tensor, gt_tensor, 100)
    # triplet_loss_OHEM(prediction_tensor, gt_tensor, feature_map_tensor)
    # grad_s = tfe.gradients_function(triplet_loss_OHEM)
    # print('gradient of square:', grad_s(prediction_tensor, gt_tensor, feature_map_tensor))

    # success()
    failed()

