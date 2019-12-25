import tensorflow as tf


def xyz_to_uvd(xyz, cam):
    """Convert points from camera to image frame.

    Arguments
    ---------
    xyz : (b, n, 3); optional b
        camera coordinates
    cam : (b, 4); optional b
        camera parameters [fx, fy, cx, cy]

    Returns
    -------
    uvd : same shape as `xyz`
        image coordinates. Note d is same as z.
    """
    uv = xyz[..., :2] * cam[..., tf.newaxis, :2] \
        / xyz[..., 2:3] + cam[..., tf.newaxis, 2:]
    uvd = tf.concat([uv, xyz[..., 2:]], axis=-1)

    return uvd


def uvd_to_xyz(uvd, cam):
    """Convert points from image to camera frame.

    Arguments
    ---------
    uvd : (b, n, 3); optional b
        image coordinates
    cam : (b, 4); optional b
        camera parameters [fx, fy, cx, cy]

    Returns
    -------
    xyz : same shape as `uvd`
        camera coordinates. Note z is same as d.
    """
    xy = (uvd[..., :2] - cam[..., tf.newaxis, 2:]) \
        * uvd[..., 2:] / cam[..., tf.newaxis, :2]
    xyz = tf.concat([xy, uvd[..., 2:]], axis=-1)

    return xyz


def depth_to_uvd(depth):
    """Returns uvd points from depth.

    Arguments
    ---------
    depth : shape=(h, w)
        depth image

    Returns
    -------
    uvd : shape=(h*w, 3)
        uvd points of corresponding depth
    """
    coords_v = tf.range(depth.shape[0], dtype=tf.float32)
    coords_u = tf.range(depth.shape[1], dtype=tf.float32)
    coords_V, coords_U = tf.meshgrid(coords_v, coords_u, indexing="ij")

    uvd = tf.reshape(
        tf.stack([coords_U, coords_V, depth], axis=2),
        [-1, 3]
    )

    return uvd


def depth_to_xyz(depth, cam):
    """Returns xyz points from depth.

    Arguments
    ---------
    depth : shape=(h, w)
        depth image

    cam : (4,)
        camera parameters [fx, fy, cx, cy]

    Returns
    -------
    xyz : shape=(h*w, 3)
        xyz points of corresponding depth
    """
    depth_uvd = depth_to_uvd(depth)
    depth_xyz = uvd_to_xyz(depth_uvd, cam)

    return depth_xyz
