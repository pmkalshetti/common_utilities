import open3d as o3d
import tensorflow as tf
import numpy as np


class Mesh:
    """Mesh wrapper for o3d.TriangleMesh.

    Attributes
    ----------
    mesh : `open3d.geometry.TriangleMesh`
        Triangle mesh object as per `open3d`.
    """

    def __init__(self, verts, triangles, color=[255, 0, 0]):
        """Creates a mesh with given vertices and triangles.

        Arguments
        ---------
        verts : np.ndarray of shape (N, 3)
            Mesh vertices.

        triangles : np.ndarray of shape (F, 3)
            Vertex indices for each triangle.

        color : array_like of shape (3,)
            RGB color triplet. color in [0, 255].
        """

        if isinstance(verts, tf.Tensor):
            verts = verts.numpy()
        if isinstance(triangles, tf.Tensor):
            triangles = triangles.numpy()

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        # self.lines = o3d.geometry.create_line_set_from_triangle_mesh(self.mesh)
        self.color = color

        self.update(verts)

    def get_verts(self):
        return np.asarray(self.vertices)

    def get_triangles(self):
        return np.asarray(self.triangles)

    def update(self, verts, update_normals=True):
        """Updates mesh vertices.

        Arguments
        ---------
        verts : np.ndarray of shape (N, 3)
            Updated mesh vertices.
        """
        if isinstance(verts, tf.Tensor):
            verts = verts.numpy()
        self.mesh.vertices = o3d.utility.Vector3dVector(verts)
        # self.lines.points = o3d.utility.Vector3dVector(verts)

        # required for rendering
        if update_normals:
            self.mesh.compute_triangle_normals()
            self.mesh.compute_vertex_normals()

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
        self.mesh.paint_uniform_color(color)

    def write(self, path):
        """Writes mesh to file as vertices and faces.

        Arguments
        ----------
        path : string
            Path to output file.
        """
        assert not tf.io.gfile.exists(path), "File already exists."

        with open(path, "w") as file:
            for v in self.get_verts():
                file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            for f in self.get_triangles():
                file.write("f {:d} {:d} {:d}\n".format(f[0], f[1], f[2]))
