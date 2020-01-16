import os
import numpy as np
import open3d as o3d


class Visualizer:
    """Custom visualizer class.

    Attributes
    ----------
    vis : o3d.Visualizer
    """

    def __init__(self, window_name="Visualizer", left=50, top=50,
                 width=640, height=480, fx=475, fy=475, pos_cam=[0, 0, 0],
                 background_color=[1.0, 1.0, 1.0],
                 point_size=5.0,
                 mesh_show_wireframe=True, mesh_shade_option=0):
        """Creates a visualizer with given properties.

        Arguments
        ---------
        window_name : string
            Name on title bar.

        left, top : int
            Top-left corner position of window.

        width, height : int
            Dimensions of window.

        fx, fy : float
            Focal lengths in pixel.

        pos_cam : list of float
            3D position of camera.
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.create_window(window_name, width, height, left, top)
        self.set_render_option(
            background_color, point_size,
            mesh_show_wireframe, mesh_shade_option
        )
        self.set_view(
            width, height,
            fx, fy,
            width/2 - 0.5, height/2 - 0.5,
            pos_cam
        )

    def create_window(self, window_name, width, height, left, top):
        cwd = os.getcwd()  # to handle issue on Mac
        self.vis.create_window(
            window_name=window_name,
            width=width, height=height, left=left, top=top
        )
        os.chdir(cwd)  # to handle issue on Mac

    def set_render_option(self, background_color,
                          point_size,
                          mesh_show_wireframe, mesh_shade_option):
        render_option = self.vis.get_render_option()
        render_option.background_color = background_color
        render_option.point_size = point_size
        render_option.mesh_show_wireframe = mesh_show_wireframe
        # render_option.mesh_shade_option = mesh_shade_option
        # render_option.load_from_json(path_render_option)

    def set_view(self, width, height, fx, fy, cx, cy, pos_cam):
        """Sets camera view."""
        view_control = self.vis.get_view_control()
        view_control.set_constant_z_far(3000)
        view_control.scale(1)

        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        camera_extrinsic = np.array([
            [1, 0, 0, pos_cam[0]],
            [0, 1, 0, pos_cam[1]],
            [0, 0, 1, pos_cam[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        pinhole_camera_parameters = o3d.camera.PinholeCameraParameters()
        pinhole_camera_parameters.intrinsic = camera_intrinsic
        pinhole_camera_parameters.extrinsic = camera_extrinsic
        view_control.convert_from_pinhole_camera_parameters(
            pinhole_camera_parameters
        )

    def add_mesh(self, mesh):
        """Add a mesh to the visualizer.

        Arguments
        ---------
        mesh : `lib.o3d_wrapper.Mesh` object
        """
        self.vis.add_geometry(mesh.mesh)

    def add_pcd(self, pc):
        """Add a pcd to the visualizer.

        Arguments
        ---------
        pc : `lib.o3d_wrapper.PointCloud` object
        """
        self.vis.add_geometry(pc.pcd)

    def add_lineset(self, lineset):
        """Add lineset to visualizer.
        
        Arguments
        ---------
        lineset : `lib.o3d_wrapper.Lineset` object.
        """
        self.vis.add_geometry(lineset.lineset)

    def show_frame(self, pos=np.array([0, 0, 0]), scale=100):
        """Adds a coordinate frame in visualizer."""
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=scale, origin=pos
        )
        self.vis.add_geometry(frame)

    def update(self, geometries):
        if not isinstance(geometries, list):
            geometries = [geometries]

        [self.vis.update_geometry(geometry) for geometry in geometries]

    def show(self):
        """Updates geometries in visualizer.

        Returns
        -------
        open_window : bool
            `False` if window is to be closed.
        """
        # self.vis.update_geometry()
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
        img = (img * 255).astype(np.uint8)

        return img

    def get_cam(self):
        pinhole_camera_parameters = \
            self.vis.convert_to_pinhole_camera_parameters()
        camera_intrinsic = pinhole_camera_parameters.intrinsic
        camera_extrinsic = pinhole_camera_parameters.extrinsic

        fx, fy = camera_intrinsic.get_focal_length()
        cx, cy = camera_intrinsic.get_principal_point()

        return np.stack((fx, fy, cx, cy))

    def __del__(self):
        self.vis.destroy_window()

    # used for debugging
    def save_img(self, path):
        """Saves visualizer window as uint8 image."""
        self.vis.capture_screen_image(path, True)


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.show_frame()
    while True:
        if not visualizer.show():
            break
