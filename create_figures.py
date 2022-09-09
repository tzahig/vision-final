from vanishing import draw_lines
from calibrated_reconstruction import run_reconstruction
from visualize import visualize

import os
import numpy as np

FIGURES_DIR = 'figures'
USE_CACHE = False


def main():
    if not os.path.isdir(FIGURES_DIR):
        os.mkdir(FIGURES_DIR)

    datasets = (
        ('observatory', ('DSC_1000.JPG', 'DSC_0996.JPG')),
        ('living_room', ('DSC_1586.JPG', 'DSC_1587.JPG')),
        # ('entrance', ('tzahi_door1.jpg', 'tzahi_door2.jpg')),
    )

    for scene_name, img_names in datasets:
        create_figures_for_scene(scene_name, img_names)


def create_figures_for_scene(scene_name, img_names):
    # Vanishing lines / points
    for i, img_name in enumerate(img_names):
        if scene_name in ['entrance']:
            fn = f'{scene_name}/{img_name}'
        else:
            fn = f'data/{scene_name}/images/dslr_images_undistorted/{img_name}'
        draw_lines(
            fn,
            False,
            f'{FIGURES_DIR}/vanishing_lines_{scene_name}_{i + 1}.png',
        )
        draw_lines(
            fn,
            True,
            f'{FIGURES_DIR}/vanishing_points_{scene_name}_{i + 1}.png',
        )

    open3d_config_path = (
        f'open3d_configs/{scene_name}_{img_names[0]}_{img_names[1]}.json'
    )

    if not os.path.isfile(open3d_config_path):
        print(f'{open3d_config_path} for initial view does not exist.')
        print('In the viewer, press p to create it for the first time.')
        print('This will save a json. Rename it to above file.')
        open3d_config_path = None

    point_cloud_path = f'cloud_{scene_name}.npy'

    if USE_CACHE and os.path.isfile(point_cloud_path):
        cloud, colors = np.load(point_cloud_path)
    else:
        cloud, colors = run_reconstruction(scene_name, img_names)
        np.save(point_cloud_path, (cloud, colors))

    # Using initial view from configuration,
    # rotate the camera and save some views:
    rotation_xs = 3
    rotation_ys = 6
    rotation_distance_delta = 100.0

    for i in range(rotation_xs):
        out_path = f'{FIGURES_DIR}/{scene_name}_reconstruction_{i + 1}.png'
        visualize(
            cloud,
            colors,
            open3d_config_path,
            out_path,
            rotation_x=(i * rotation_distance_delta),
        )
    for i in range(rotation_ys):
        out_path = f'{FIGURES_DIR}/{scene_name}_reconstruction_{i + 1 + rotation_xs}.png'
        visualize(
            cloud,
            colors,
            open3d_config_path,
            out_path,
            rotation_y=(i * rotation_distance_delta),
        )


if __name__ == '__main__':
    main()
