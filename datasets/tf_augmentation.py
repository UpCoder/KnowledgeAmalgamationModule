import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables


def _assert(cond, ex_type, msg):
    """A polymorphic assert, works with tensors and boolean expressions.
    If `cond` is not a tensor, behave like an ordinary assert statement, except
    that a empty list is returned. If `cond` is a tensor, return a list
    containing a single TensorFlow assert op.
    Args:
      cond: Something evaluates to a boolean value. May be a tensor.
      ex_type: The exception class to use.
      msg: The error message.
    Returns:
      A list, containing at most one assert op.
    """
    if _is_tensor(cond):
        return [control_flow_ops.Assert(cond, [msg])]
    else:
        if not cond:
            raise ex_type(msg)
        else:
            return []


def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.
    Args:
      x: A python object to check.
    Returns:
      `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
    return isinstance(x, (ops.Tensor, variables.Variable))


def _ImageDimensions(image):
    """Returns the dimensions of an image tensor.
    Args:
      image: A 3-D Tensor of shape `[height, width, channels]`.
    Returns:
      A list of `[height, width, channels]` corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def _Check3DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.
    Args:
      image: 3-D Tensor of shape [height, width, channels]
        require_static: If `True`, requires that all dimensions of `image` are
        known and non-zero.
    Raises:
      ValueError: if `image.shape` is not a 3-vector.
    Returns:
      An empty list, if `image` has fully defined dimensions. Otherwise, a list
        containing an assert op is returned.
    """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' must be three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' must be fully defined.")
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image),
                                          ["all dims of 'image.shape' "
                                           "must be > 0."])]
    else:
        return []


def random_rotate90(image, output_mask=None):
    with tf.name_scope('random_rotate90'):
        k = random_ops.random_uniform([], 0, 10000)
        k = tf.cast(k, tf.int32)

        image_shape = tf.shape(image)
        image = tf.image.rot90(image, k=k)
        if output_mask is not None:
            output_mask = tf.image.rot90(output_mask, k=k)
        return image, output_mask


def flip_vertical(image, output_mask=None):
    with tf.name_scope('flip_vertical'):
        image = tf.image.flip_up_down(image)
        if output_mask is not None:
            output_mask = tf.image.flip_up_down(output_mask)
        return image, output_mask


def flip_horizontal(image, output_mask=None):
    with tf.name_scope('flip_horizontal'):
        image = tf.image.flip_left_right(image)
        if output_mask is not None:
            output_mask = tf.image.flip_left_right(output_mask)
        return image, output_mask


def resize_image_bboxes_with_crop_or_pad(image, target_height, target_width, mask_image=None):
    """Crops and/or pads an image to a target width and height.
    Resizes an image to a target width and height by either centrally
    cropping the image or padding it evenly with zeros.
    If `width` or `height` is greater than the specified `target_width` or
    `target_height` respectively, this op centrally crops along that dimension.
    If `width` or `height` is smaller than the specified `target_width` or
    `target_height` respectively, this op centrally pads with 0 along that
    dimension.
    Args:
      image: 3-D tensor of shape `[height, width, channels]`
      target_height: Target height.
      target_width: Target width.
    Raises:
      ValueError: if `target_height` or `target_width` are zero or negative.
    Returns:
      Cropped and/or padded image of shape
        `[target_height, target_width, channels]`
    """
    with tf.name_scope('resize_with_crop_or_pad'):
        image = ops.convert_to_tensor(image, name='image')
        if mask_image is not None:
            print('Image: ', image)
            print('MaskImage: ', mask_image)
            mask_image = ops.convert_to_tensor(mask_image, name='image')

        assert_ops = []
        assert_ops += _Check3DImage(image, require_static=False)
        assert_ops += _assert(target_width > 0, ValueError,
                              'target_width must be > 0.')
        assert_ops += _assert(target_height > 0, ValueError,
                              'target_height must be > 0.')

        image = control_flow_ops.with_dependencies(assert_ops, image)

        # `crop_to_bounding_box` and `pad_to_bounding_box` have their own checks.
        # Make sure our checks come first, so that error messages are clearer.
        if _is_tensor(target_height):
            target_height = control_flow_ops.with_dependencies(
                assert_ops, target_height)
        if _is_tensor(target_width):
            target_width = control_flow_ops.with_dependencies(assert_ops, target_width)

        def max_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.maximum(x, y)
            else:
                return max(x, y)

        def min_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.minimum(x, y)
            else:
                return min(x, y)

        def equal_(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return math_ops.equal(x, y)
            else:
                return x == y

        height, width, _ = _ImageDimensions(image)
        width_diff = target_width - width
        offset_crop_width = max_(-width_diff // 2, 0)
        offset_pad_width = max_(width_diff // 2, 0)

        height_diff = target_height - height
        offset_crop_height = max_(-height_diff // 2, 0)
        offset_pad_height = max_(height_diff // 2, 0)

        # Maybe crop if needed.
        height_crop = min_(target_height, height)
        width_crop = min_(target_width, width)
        cropped = tf.image.crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                                height_crop, width_crop)

        # Maybe pad if needed.
        resized = tf.image.pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                               target_height, target_width)


        # In theory all the checks below are redundant.
        if resized.get_shape().ndims is None:
            raise ValueError('resized contains no shape.')

        resized_height, resized_width, _ = _ImageDimensions(resized)

        assert_ops = []
        assert_ops += _assert(equal_(resized_height, target_height), ValueError,
                              'resized height is not correct.')
        assert_ops += _assert(equal_(resized_width, target_width), ValueError,
                              'resized width is not correct.')

        resized = control_flow_ops.with_dependencies(assert_ops, resized)
        return resized


def distort_color(image, color_ordering=0, fast_mode=True, scope=None, divided=255.):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / divided)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / divided)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / divided)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / divided)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / divided)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / divided)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        # return tf.clip_by_value(image, 0.0, 1.0)
        return image


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.
    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]


def get_aug_tensor(output_image, output_mask, prob=0.6):
    '''
    输入的tensortensor is uint8
    :param image:
    :param prob:
    :return: tranformed image and transformed mask
    '''
    # rotate aug
    output_image = tf.cast(output_image, tf.float32)
    # rnd = tf.random_uniform((), minval=0, maxval=1)
    # def rotate():
    #     return random_rotate90(output_image, output_mask)
    #
    # def non_op():
    #     return output_image, output_mask
    # output_image, output_mask = tf.cond(tf.less(rnd, prob), rotate, non_op)
    #
    # if output_image.dtype != tf.float32:
    #     output_image = tf.image.convert_image_dtype(output_image, dtype=tf.float32)
    #
    # # flip horizontal
    # def flip_h():
    #     return flip_horizontal(output_image, output_mask)
    #
    # def non_flip_h():
    #     return output_image, output_mask
    # rnd = tf.random_uniform((), minval=0, maxval=1)
    # output_image, output_mask = tf.cond(tf.less(rnd, prob), flip_h, non_flip_h)
    #
    # # flip horizontal
    # def flip_v():
    #     return flip_vertical(output_image, output_mask)
    #
    # def non_flip_v():
    #     return output_image, output_mask
    # rnd = tf.random_uniform((), minval=0, maxval=1)
    # output_image, output_mask = tf.cond(tf.less(rnd, prob), flip_v, non_flip_v)

    # color augm
    rnd3 = tf.random_uniform((), minval=0, maxval=1)

    def color_aug():
        tmp_img = apply_with_random_selector(
            output_image,
            lambda x, ordering: distort_color(x, ordering, True, divided=3.),
            num_cases=4)
        tmp_img = tf.cast(tmp_img, tf.float32)
        return tmp_img

    def non_color_aug():
        tmp_img = tf.cast(output_image, tf.float32)
        return tmp_img
    output_image = tf.cond(tf.less(rnd3, prob), color_aug, non_color_aug)

    # add noise
    rnd4 = tf.random_uniform((), minval=0, maxval=1)

    def noise_aug():
        tmp_image = tf.cast(output_image, tf.float32)
        noise_mask_tensor = tf.random_normal(shape=tf.shape(tmp_image), mean=0.0, stddev=10,
                                             dtype=tf.float32)
        tmp_image += noise_mask_tensor
        tmp_image = tf.cast(tmp_image, tf.float32)
        return tmp_image

    def non_noise_aug():
        return output_image
    output_image = tf.cond(tf.less(rnd4, prob), noise_aug, non_noise_aug)

    rnd5 = tf.random_uniform((), minval=0, maxval=1)

    def noise_aug_2():
        tmp_image = tf.cast(output_image, tf.float32)
        noise_mask_tensor = tf.random_normal(shape=(), mean=0.0, stddev=10,
                                             dtype=tf.float32)
        tmp_image += noise_mask_tensor
        tmp_image = tf.cast(tmp_image, tf.float32)
        return tmp_image

    def non_noise_aug_2():
        return output_image
    output_image = tf.cond(tf.less(rnd5, prob), noise_aug_2, non_noise_aug_2)

    rnd6 = tf.random_uniform((), minval=0, maxval=1)

    def noise_aug_3():
        tmp_image = tf.cast(output_image, tf.float32)
        noise_mask_tensor = tf.random_normal(shape=(), mean=0.0, stddev=10,
                                             dtype=tf.float32)
        tmp_image -= noise_mask_tensor
        tmp_image = tf.cast(tmp_image, tf.float32)
        return tmp_image

    def non_noise_aug_3():
        return output_image

    output_image = tf.cond(tf.less(rnd6, prob), noise_aug_3, non_noise_aug_3)

    return output_image, output_mask


if __name__ == '__main__':
    print('')
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input_image_tensor = tf.placeholder(tf.uint8, shape=[224, 224, 3])
    # input_mask_tensor = tf.ones_like(input_image_tensor)
    # output_image_tensor, output_mask_tensor = get_aug_tensor(input_image_tensor, 0)
    # sess = tf.Session()
    # input_path = './tmp/augmentation/tmp-01.jpg'
    # import cv2
    # import numpy as np
    # img = cv2.imread(input_path)
    # if img is None:
    #     assert False
    # img = cv2.resize(img, (224, 224))
    # for i in range(20):
    #     output_image, output_mask = sess.run([output_image_tensor, output_mask_tensor], feed_dict={
    #         input_image_tensor: img
    #     })
    #     print(np.shape(output_image))
    #     cv2.imwrite('./tmp/augmentation/tmp-01-aug-{}.jpg'.format(i), output_image)
    #     cv2.imwrite('./tmp/augmentation/tmp-01-aug-{}_mask.jpg'.format(i), np.asarray(output_mask * 255., np.uint8))