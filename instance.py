import tensorflow as tf


def tf2np(tensor):
    """Converts tensorflow tensor to numpy if not numpy."""
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()
    return tensor


def make_list(obj):
    """Makes list of obj if obj is not a list already."""
    if not isinstance(obj, list):
        obj = [obj]

    return obj
