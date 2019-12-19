import tensorflow as tf
import numpy as np


def rot_mat_x(angle):
    """Returns 3x3 rotation matrix about X axis.

    Arguments
    ---------
    angle : float
        Angle in radian.

    Returns
    -------
    rot_mat : tf.Tensor of shape (3, 3)
        Rotation matrix corresponding to angle about X axis.
    """
    row1 = tf.stack([1., 0,             0])
    row2 = tf.stack([0,  tf.cos(angle), -tf.sin(angle)])
    row3 = tf.stack([0,  tf.sin(angle), tf.cos(angle)])
    rot_mat = tf.stack([row1, row2, row3], axis=0)

    return rot_mat


def rot_mat_y(angle):
    """Returns 3x3 rotation matrix about Y axis.

    Arguments
    ---------
    angle : float
        Angle in radian.

    Returns
    -------
    rot_mat : tf.Tensor of shape (3, 3)
        Rotation matrix corresponding to angle about Y axis.
    """
    row1 = tf.stack([tf.cos(angle), 0, tf.sin(angle)])
    row2 = tf.stack([0.,            1, 0])
    row3 = tf.stack([-tf.sin(angle), 0, tf.cos(angle)])
    rot_mat = tf.stack([row1, row2, row3], axis=0)

    return rot_mat


def rot_mat_z(angle):
    """Returns 3x3 rotation matrix about Z axis.

    Arguments
    ---------
    angle : float
        Angle in radian.

    Returns
    -------
    rot_mat : tf.Tensor of shape (3, 3)
        Rotation matrix corresponding to angle about Z axis.
    """
    row1 = tf.stack([tf.cos(angle), -tf.sin(angle), 0])
    row2 = tf.stack([tf.sin(angle), tf.cos(angle),  0])
    row3 = tf.stack([0.,             0,             1])
    rot_mat = tf.stack([row1, row2, row3], axis=0)

    return rot_mat
