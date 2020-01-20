import open3d as o3d
import numpy as np
import tensorflow as tf


class PointCloud:
    """Point Cloud wrapper for open3d.PointCloud.

    Attributes
    ----------
    pcd : `open3d.geometry.PointCloud`
        Point cloud object as per `open3d`.

    color : array_like of shape (3,)
        RGB color triplet. color in [0, 255].
    """

    def __init__(self, pts=None, color=[255, 0, 0]):
        """Creates a point cloud from given xyz points.

        Arguments
        ---------
        pts : np.ndarray of shape(n_points, 3)
            3D coordinates of points in point cloud.

        color : array_like of shape (3,)
            RGB color triplet. color in [0, 255].
        """
        if isinstance(pts, tf.Tensor):
            pts = pts.numpy()

        self.pcd = o3d.geometry.PointCloud()
        self.color = color

        if pts is not None:
            self.update(pts)

    def set_normals(self, normals):
        if isinstance(normals, tf.Tensor):
            normals = normals.numpy()

        self.pcd.normals = o3d.utility.Vector3dVector(normals)

    def update(self, pts):
        """Updates location of points.

        Arguments
        ---------
        pts : np.ndarray of shape(n_points, 3)
            3D coordinates of points in point cloud.
        """
        if isinstance(pts, tf.Tensor):
            pts = pts.numpy()

        self.pcd.points = o3d.utility.Vector3dVector(pts)
        self.set_color(self.color)

    def set_color(self, color):
        """Color point cloud.

        Arguments
        ---------
        color : array_like of shape (3,)
            RGB color triplet. color in [0, 255].
        """
        self.color = color
        color = [c/255 for c in color]
        self.pcd.paint_uniform_color(color)

    def crop(self, min_bound, max_bound):
        """Crops point cloud.

        Arguments
        ---------
        min_bound : np.array of shape (3,)
            Coordinates for minimum clipping.

        max_bound : np.array of shape (3,)
            Coordinates for maximum clipping.
        """
        self.pcd = o3d.geometry.crop_point_cloud(
            self.pcd, min_bound, max_bound)

    def recenter(self, center=None):
        """Subtract `center` from all points.

        Arguments
        ---------
        center : np.array of shape (3,)
            Subtract this coordinate from all points.
        """
        pts = np.asarray(self.pcd.points)

        if center is None:
            center = np.mean(pts, axis=0)

        pts -= center
        self.update(pts)

    def remove_outliers(self, n_pts=20, radius=15):
        """Removes outliers from point cloud.

        Arguments
        ---------
        n_pts : int
            Threshold on number of points in neighbour.
        radius : float
            Radius of neighbourhood in mm.
        """
        self.pcd, inds = self.pcd.remove_radius_outlier(n_pts, radius)

    def estimate_normals(self):
        self.pcd.estimate_normals()
        self.pcd.orient_normals_towards_camera_location()

        return np.asarray(self.pcd.normals).astype(np.float32)
