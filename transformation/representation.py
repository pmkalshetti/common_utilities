import tensorflow as tf
from tensorflow_graphics.geometry import transformation


def axis_angle2rot_mat(axis_angle):
    """Converts axis angle representation to rotation matrix.

    Note: The `n_joints` axis is optional.

    Arguments
    ----------
    axis_angle : tf.Tensor of shape (`n_joints`, 3)
        Rotation at each joint represented as axis angle.

    Returns
    -------
    mat_rot : tf.Tensor of shape (`n_joints`, 3, 3)
        Rotation matrix corresponding to each joint.
    """
    axis = tf.math.l2_normalize(axis_angle, axis=-1)
    angle = tf.norm(axis_angle, axis=-1, keepdims=True)
    mat_rot = transformation.rotation_matrix_3d.from_axis_angle(
        axis, angle)

    return mat_rot


def rot_mat2axis_angle(rot_mat):
    """Converts rotation matrix to axis angle representation.

    Note: batch supported.

    Arguments
    ---------
    rot_mat : tf.Tensor of shape (3,3)
        Rotation matrix.

    Returns
    -------
    axis_angle : tf.Tensor of shape (3,)
        Axis angle representation of `rot_mat`.
    """
    axis, angle = transformation.axis_angle.from_rotation_matrix(rot_mat)
    axis_angle = axis * angle

    return axis_angle
