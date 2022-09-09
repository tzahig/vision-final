import numpy as np
import numpy.typing as npt


def visualize(
    points: npt.ArrayLike,
    colors: npt.ArrayLike,
    open3d_config_path: str = None,
    out_path: str = None,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
):
    """
    Visualize a point cloud
    :param points: The points to visualize as a numpy array of shape (N, 3)
    :param colors: The colors of each of the points as a numpy array of shape (N, 3)
    :param open3d_config_path json path for camera config
    :param out_path will save visualization to file if is not None
    :param rotation_x: x argument to open3d.visualization.ViewControl.rotate()
    :param rotation_y: y argument to open3d.visualization.ViewControl.rotate()
    """
    import open3d as o3d

    # Build point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255)

    # create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True

    render_option.point_size = 4
    vis.add_geometry(pcd)

    # Set camera
    ctr = vis.get_view_control()
    if open3d_config_path is not None:
        parameters = o3d.io.read_pinhole_camera_parameters(open3d_config_path)
        ctr.convert_from_pinhole_camera_parameters(parameters)

    # rotate from initial view:
    ctr.rotate(rotation_x, rotation_y)

    opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])

    vis.run()
    if out_path is not None:
        vis.capture_screen_image(out_path, do_render=True)
    vis.destroy_window()


def main():
    points = np.load('cloud_t.npy')
    print(points.shape)
    visualize(points, None)


if __name__ == '__main__':
    main()
