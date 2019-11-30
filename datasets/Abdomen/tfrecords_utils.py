import tensorflow as tf
import os
import numpy as np


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class Example:
    def __init__(self):
        self.attribute_names = []
        self.attribute_values = []
        self.attribute_types = []

    def init_values(self, names, types, values):
        if len(names) != len(types) and len(names) != len(values):
            assert False
        self.attribute_names = names
        self.attribute_types = types
        self.attribute_values = values

    def get_examples(self):
        features = {}
        for attribute_name, attribute_type, attribute_value in zip(self.attribute_names,
                                                                   self.attribute_types, self.attribute_values):
            if attribute_type == 'int':
                features[attribute_name] = _int64_feature(attribute_value)
            elif attribute_type == 'np_array':
                features[attribute_name] = _int64_feature(attribute_value.flatten().tolist())
            elif attribute_type == 'str':
                features[attribute_name] = _bytes_feature(attribute_value)
            else:
                assert False
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example