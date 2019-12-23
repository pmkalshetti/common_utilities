import os
import numpy as np
import open3d as o3d
import tensorflow as tf


class Visualizer:
    """Custom visualizer class.

    Attributes
    ----------
    vis : o3d.Visualizer

    width, height : int
        Dimensions of window.

    fx, fy : float
        Focal lengths in pixel.
    cx, cy : float
        Camera principal point.
    pos_cam : list of float
        3D position of camera.

    view_control : o3d.Visualizer.View_control
    """

    def __init__(self, window_name="Visualizer", width=640, height=480,
                 left=50, top=50, path_render_option=None, fx=475, fy=475,
                 pos_cam=[0, 0, 0]):
        """Creates a visualizer with given properties.

        Arguments
        ---------
        window_name : string
            Name on title bar.

        width, height : int
            Dimensions of window.

        left, top : int
            Top-left corner position of window.

        path_render_option : string
            Path to json file containing render options for visualizer.

        fx, fy : float
            Focal lengths in pixel.

        pos_cam : list of float
            3D position of camera.
        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = self.width/2 - 0.5
        self.cy = self.height/2 - 0.5
        self.pos_cam = pos_cam

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        cwd = os.getcwd()  # to handle issue on Mac
        self.vis.create_window(window_name=window_name,
                               width=width,
                               height=height,
                               left=left,
                               top=top)
        os.chdir(cwd)  # to handle issue on Mac

        if path_render_option is not None:
            render_option = self.vis.get_render_option()
            render_option.load_from_json(path_render_option)

        self.view_control = self.vis.get_view_control()
        self.set_view()

    def set_view(self):
        """Sets camera view."""
        self.view_control.scale(1)
        camera_intrinsic = o3d.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        camera_extrinsic = np.array([
            [1, 0, 0, self.pos_cam[0]],
            [0, 1, 0, self.pos_cam[1]],
            [0, 0, 1, self.pos_cam[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        pinhole_camera_parameters = o3d.PinholeCameraParameters()
        pinhole_camera_parameters.intrinsic = camera_intrinsic
        pinhole_camera_parameters.extrinsic = camera_extrinsic
        self.view_control.convert_from_pinhole_camera_parameters(
            pinhole_camera_parameters)

    def add_mesh(self, mesh):
        """Add a mesh to the visualizer.

        Arguments
        ---------
        mesh : `lib.o3d_wrapper.Mesh` object
        """
        self.vis.add_geometry(mesh.mesh)
        self.set_view()

    def add_pcd(self, pc):
        """Add a pcd to the visualizer.

        Arguments
        ---------
        pc : `lib.o3d_wrapper.PointCloud` object
        """
        self.vis.add_geometry(pc.pcd)
        self.set_view()

    def add_lineset(self, lineset):
        """Add lineset to visualizer.
        
        Arguments
        ---------
        lineset : `lib.o3d_wrapper.Lineset` object.
        """
        self.vis.add_geometry(lineset.lineset)
        self.set_view()

    def show_frame(self, pos=np.array([0, 0, 0]), scale=100):
        """Adds a coordinate frame in visualizer."""
        frame = o3d.geometry.create_mesh_coordinate_frame(
            size=scale, origin=pos)
        self.vis.add_geometry(frame)
        self.set_view()

    def show(self):
        """Updates geometries in visualizer.

        Returns
        -------
        open_window : bool
            `False` if window is to be closed.
        """
        self.vis.update_geometry()
        open_window = self.vis.poll_events()

        return open_window

    def run(self):
        while True:
            if not self.show():
                break

    def reset_view(self):
        """Resets view point. Useful after adding new geometries."""
        self.vis.reset_view_point(True)

    def depth_buffer(self):
        """Returns depth buffer.

        Returns
        -------
        depth : np.ndarray of shape (self.height, self.width)
            Depth buffer of vis.
        """
        depth = self.vis.capture_depth_float_buffer(True)
        depth = np.asarray(depth)

        return depth

    def screen_buffer(self):
        """Returns screen buffer.

        Returns
        -------
        img : np.ndarray of shape (self.height, self.width)
            Screen buffer of vis.
        """
        img = self.vis.capture_screen_float_buffer(True)
        img = np.asarray(img)

        return img

    def get_cam(self):
        return tf.constant((self.fx, self.fy, self.cx, self.cy))

    def __del__(self):
        self.vis.destroy_window()

    # used for debugging
    def save_img(self, path):
        """Saves visualizer window as uint8 image."""
        self.vis.capture_screen_image(path, True)
