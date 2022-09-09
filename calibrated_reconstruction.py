import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import cholesky, inv
from skimage import feature

from iac_solver import omega_from_orthogonal_vanishing_points
from reconstruct_using_calibration import reconstruct_using_calibration
from vanishing import get_vanishing
from visualize import visualize

from PIL import Image

np.set_printoptions(precision=3, suppress=True)


def get_intrinsics(scene_name):
    cameras_path = (
        f'data/{scene_name}/dslr_calibration_undistorted/cameras.txt'
    )
    lines = open(cameras_path).readlines()
    for line in lines:
        if line.startswith('0'):
            break
    line = line.split(' ')[4:]
    fx, fy, cx, cy = [float(x) for x in line]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def k_from_omega(omega):
    k = inv(cholesky(omega)).T
    k /= k[2, 2]
    return k


def plot_matches(img1, img2, pts1, pts2):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot()

    n = len(pts1)
    assert n == len(pts2)
    matches = np.vstack([np.arange(n), np.arange(n)]).T
    feature.plot_matches(
        ax, img1, img2, pts1[:, ::-1], pts2[:, ::-1], matches=matches
    )
    ax.axis('off')
    ax.set_title(f'Matches for all channels')
    plt.show()


def run_reconstruction(
        scene_name,
        img_names,
        debug_matches=False,
        use_gt_k=False,
):
    if scene_name in ['entrance']:
        img_dir = f'{scene_name}'
        gt_exists = False
    else:
        img_dir = f'data/{scene_name}/images/dslr_images_undistorted'
        gt_exists = True
    img_paths = [f'{img_dir}/{p}' for p in img_names]

    if use_gt_k:
        assert gt_exists
        k = get_intrinsics(scene_name)
        k1 = k
        k2 = k
    else:
        if gt_exists:
            k = get_intrinsics(scene_name)
            print('k_gt')
            print(k)
        v1 = get_vanishing(img_paths[0])
        v2 = get_vanishing(img_paths[1])

        aspect_ratio = 1.
        k1 = k_from_omega(omega_from_orthogonal_vanishing_points(v1, aspect_ratio))
        k2 = k_from_omega(omega_from_orthogonal_vanishing_points(v2, aspect_ratio))

        print('k1')
        print(k1)
        print('k2')
        print(k2)

    for p in img_paths:
        assert os.path.isfile(p), f'{p} not found'
    imgs = [cv.imread(p, 0) for p in img_paths]

    kp1, desc1 = detector_descriptor(imgs[0])
    kp2, desc2 = detector_descriptor(imgs[1])

    fundamental_mat, pts1, pts2 = find_fundamental(desc2, kp2, desc1, kp1)

    # Read colors
    imgs = [np.asarray(Image.open(p)) for p in img_paths]

    if debug_matches:
        plot_matches(imgs[0], imgs[1], pts1, pts2)

    points_3d, colors = reconstruction_with_calibration(
        fundamental_mat, imgs, pts1, pts2, k1, k2
    )
    return points_3d, colors


def reconstruction_with_calibration(fundamental_mat, imgs, pts1, pts2, k1, k2):
    # Use own implementation of recoverPose and triangulate:
    points_3d = reconstruct_using_calibration(pts1, pts2, fundamental_mat, k1, k2)

    # essential_mat = k2.T @ fundamental_mat @ k1
    # _, r, t, _ = cv.recoverPose(essential_mat, pts1, pts2, k)
    # rt0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    # rt1 = np.empty((3, 4))
    # rt1[:3, :3] = r @ rt0[:3, :3]
    # rt1[:3, 3] = rt0[:3, 3] + (rt0[:3, :3] @ t.ravel())
    #
    # p1 = k1 @ rt0
    # p2 = k2 @ rt1
    #
    # points_3d = cv.triangulatePoints(p1, p2, pts1.T, pts2.T)
    # points_3d /= points_3d[3]
    #
    # points_3d = points_3d.T[:, :3]

    colors = np.array([imgs[0][p[1], p[0], :] for p in pts1.astype(int)])
    return points_3d, colors


def find_fundamental(desc2, kp2, desc1, kp1):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    fundamental_mat, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return fundamental_mat, pts1, pts2


def detector_descriptor(img):
    sift = cv.SIFT_create()
    return sift.detectAndCompute(img, None)


def main():
    # scene_name = 'observatory'
    # img_names = 'DSC_1003.JPG', 'DSC_1005.JPG'
    # img_names = 'DSC_1000.JPG', 'DSC_0996.JPG'

    scene_name = 'living_room'
    # img_names = 'DSC_1566.JPG', 'DSC_1567.JPG'
    # img_names = 'DSC_1568.JPG', 'DSC_1567.JPG'
    img_names = 'DSC_1586.JPG', 'DSC_1587.JPG'
    points_3d, colors = run_reconstruction(
        scene_name, img_names, debug_matches=False, use_gt_k=False
    )
    visualize(points_3d, colors)


if __name__ == '__main__':
    main()
