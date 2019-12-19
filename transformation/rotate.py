import tensorflow as tf
from .rot_mat import rot_mat_x, rot_mat_y, rot_mat_z


def rotate_x(points, angle):
    """Rotates `points` about x axis by `angle`.

    Arguments
    ---------
    points : tf.Tensor of shape (n, 3)
        3D position of points.

    angle : tf.Tensor of shape ()
        angle of rotation in radians.

    Returns
    -------
    points_rot : tf.Tensor of shape (n, 3)
        3D position of rotated points.
    """
    rot_mat = rot_mat_x(angle)
    points_rot = tf.tensordot(points, rot_mat, axes=[[1], [1]])

    return points_rot


def rotate_y(points, angle):
    """Rotates `points` about y axis by `angle`.

    Arguments
    ---------
    points : tf.Tensor of shape (n, 3)
        3D position of points.

    angle : tf.Tensor of shape ()
        angle of rotation in radians.

    Returns
    -------
    points_rot : tf.Tensor of shape (n, 3)
        3D position of rotated points.
    """
    rot_mat = rot_mat_y(angle)
    points_rot = tf.tensordot(points, rot_mat, axes=[[1], [1]])

    return points_rot


def rotate_z(points, angle):
    """Rotates `points` about z axis by `angle`.

    Arguments
    ---------
    points : tf.Tensor of shape (n, 3)
        3D position of points.

    angle : tf.Tensor of shape ()
        angle of rotation in radians.

    Returns
    -------
    points_rot : tf.Tensor of shape (n, 3)
        3D position of rotated points.
    """
    rot_mat = rot_mat_z(angle)
    points_rot = tf.tensordot(points, rot_mat, axes=[[1], [1]])

    return points_rot
