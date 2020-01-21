import numpy as np
import tensorflow as tf
from common_utilities.instance import tf2np
from tensorflow_graphics.geometry.representation.mesh.normals import \
    face_normals, vertex_normals


def compute_barycentric_coords(verts, triangles, n_samples):
    """Computes barycentric coordinates and corresponding triangle indices for
    mesh sampling.

    Arguments
    ---------
    verts : np.ndarray of shape (n_verts, 3)
        Position of mesh vertices
    triangles : np.ndarray of shape (n_triangles, 3)
        Indices of vertices that make up the triangle.
    n_samples : int
        Number of samples on mesh surface.

    Returns
    -------
    coords : np.ndarray of shape (n_samples, 2)
        Barycentric coordinates for each sampled point.
    ids_triangle : np.ndarray of shape (n_samples, )
        Index of triangle for each sampled point.
    """

    verts = tf2np(verts)
    triangles = tf2np(triangles)

    # compute area of each triangle and normalize
    cross = np.cross(
        verts[triangles[:, 0], :] - verts[triangles[:, 2], :],
        verts[triangles[:, 1], :] - verts[triangles[:, 2], :],
    )
    area_triangles = 1/2 * np.linalg.norm(cross, axis=1)
    area_normalized = area_triangles / np.sum(area_triangles)

    # sample points based on area
    n_samples_per_triangle = np.ceil(
        n_samples * area_normalized).astype(np.int32)
    n_extra = np.sum(n_samples_per_triangle) - n_samples
    if n_extra > 0:
        ids = np.nonzero(n_samples_per_triangle)[0]
        ids_extra = np.random.choice(ids, n_extra, replace=False)
        n_samples_per_triangle[ids_extra] -= 1
    n_samples = np.sum(n_samples_per_triangle)

    # map samples to triangle indices
    ids_triangle = np.zeros([n_samples, ], dtype=np.int32)
    count_samples = 0
    for idx_triangle, n_samples_this_triangle in \
            enumerate(n_samples_per_triangle):
        ids_triangle[
            count_samples:
            count_samples + n_samples_this_triangle] = idx_triangle
        count_samples += n_samples_this_triangle

    # randomly generate barycentric coordinates
    coords = np.random.rand(n_samples, 2).astype(np.float32)

    return coords, ids_triangle


def dense_sample(verts, triangles, barycentric_coords, barycentric_triangles):
    corner1_vertices = tf.gather(
        verts, triangles[barycentric_triangles, 0]
    )
    corner2_vertices = tf.gather(
        verts, triangles[barycentric_triangles, 1]
    )
    corner3_vertices = tf.gather(
        verts, triangles[barycentric_triangles, 2]
    )

    samples = (1 - tf.sqrt(barycentric_coords[:, 0:1])) \
        * corner1_vertices \
        + tf.sqrt(barycentric_coords[:, 0:1]) *\
        (1 - barycentric_coords[:, 1:]) * corner2_vertices \
        + tf.sqrt(barycentric_coords[:, 0:1]) *\
        barycentric_coords[:, 1:] * corner3_vertices

    return samples


def get_normals_at_samples(verts, triangles,
                           barycentric_coords, barycentric_triangles):
    normals_at_verts = vertex_normals(verts, triangles)
    corner1_normals = tf.gather(
        normals_at_verts, triangles[barycentric_triangles, 0]
    )
    corner2_normals = tf.gather(
        normals_at_verts, triangles[barycentric_triangles, 1]
    )
    corner3_normals = tf.gather(
        normals_at_verts, triangles[barycentric_triangles, 2]
    )
    normals_at_samples = (1 - tf.sqrt(barycentric_coords[:, 0:1])) \
        * corner1_normals \
        + tf.sqrt(barycentric_coords[:, 0:1]) *\
        (1 - barycentric_coords[:, 1:]) * corner2_normals \
        + tf.sqrt(barycentric_coords[:, 0:1]) *\
        barycentric_coords[:, 1:] * corner3_normals

    return normals_at_samples
