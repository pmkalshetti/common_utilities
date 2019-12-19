import tensorflow as tf


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # BytesList won't unpack string from EagerTensor
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def construct_proto(tensors: list, keys: list):
    """Convert data to proto string for storing as tfrecord."""
    feature = {}
    for key, tensor in zip(keys, tensors):
        feature[key] = bytes_feature(tf.io.serialize_tensor(tensor))

    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto


def parse_proto(proto, feature_description, dtypes):
    """Parses proto into data."""
    parsed = tf.io.parse_single_example(proto, feature_description)

    example = {}
    for key, dtype in zip(feature_description.keys(), dtypes):
        tensor = tf.io.parse_tensor(parsed[key], dtype)
        example[key] = tensor

    return example
