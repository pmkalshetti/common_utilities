import open3d as o3d
import tensorflow as tf


class Lineset:
    """Lineset wrapper for o3d.Lineset.

    Attributes
    ----------
    lineset : `open3d.geometry.Lineset`
        Lineset object as per `open3d`.
    """

    def __init__(self, points, lines, color=[255, 0, 0]):
        """Creates a lineset with given points and lines.

        Arguments
        ---------
        points : np.ndarray of shape (N, 3)
            points coordinates.

        lines : np.ndarray of shape (n_lines, 2)
            Lines denoted by the index of points forming the line.
        """

        if isinstance(points, tf.Tensor):
            points = points.numpy()
        if isinstance(lines, tf.Tensor):
            lines = lines.numpy()

        self.lineset = o3d.geometry.LineSet()
        self.lineset.points = o3d.utility.Vector3dVector(points)
        self.lineset.lines = o3d.utility.Vector2iVector(lines)
        colors = [color for _ in lines]
        self.lineset.colors = o3d.utility.Vector3dVector(colors)

    def update(self, points):
        """Update endpoints of lineset.

        Arguments
        ---------
        points : np.ndarray of shape (N, 3)
            points coordinates.
        """
        if isinstance(points, tf.Tensor):
            points = points.numpy()
        self.lineset.points = o3d.utility.Vector3dVector(points)
