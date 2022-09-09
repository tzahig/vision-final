import numpy as np


def triangulate(p1, pts1, p2, pts2):
    """
    :param p1: camera matrix 1
    :param pts1: image points for camera 1
    :param p2: camera matrix 2
    :param pts2: corresponding images points for camera 2
    :return: 3d points and reprojection error
    """

    def triangulate_point(i):
        """
        Each projection gives 2 equations, so we have a homogenous system with 4 equations
        and 3 unknwons.

        See: Hartley-Zisserman 12.2
        """
        x1, y1 = pts1[i, 0], pts1[i, 1]
        x2, y2 = pts2[i, 0], pts2[i, 1]
        a = np.vstack(
            [
                (x1 * p1[2, :].T - p1[0, :].T),
                (y1 * p1[2, :].T - p1[1, :].T),
                (x2 * p2[2, :].T - p2[0, :].T),
                (y2 * p2[2, :].T - p2[1, :].T),
            ]
        )
        u, s, vh = np.linalg.svd(a)

        p = vh[-1, :]
        p /= np.linalg.norm(p)
        return p[:3] / p[3]

    pts = [triangulate_point(i) for i in range(len(pts1))]

    # make homogenous 3D:
    points3d = np.hstack((np.stack(pts), np.ones((len(pts), 1))))

    def reproj_error():
        errors = []
        for i in range(len(points3d)):
            proj1 = np.dot(p1, points3d[i, :])
            proj1 = proj1[:2] / proj1[-1]

            proj2 = np.dot(p2, points3d[i, :])
            proj2 = proj2[:2] / proj2[-1]

            errors.append(
                np.sum((proj1 - pts1[i]) ** 2 + (proj2 - pts2[i]) ** 2)
            )
        return np.array(errors)

    return points3d[:, :3], reproj_error()


def m2s(essential_mat):
    """
    :param essential_mat:
    :return: four possible camera matrices for camera 2
    See 9.6.2 in HZ.
    """
    # Force matrix to be a valid essential matrix:
    # Two of its singular values identical, and last singular value 0
    u, s, v = np.linalg.svd(essential_mat)
    m = s[:2].mean()
    m_diag = np.array([[m, 0, 0], [0, m, 0], [0, 0, 0]])
    essential_mat = u @ m_diag @ v
    u, s, v = np.linalg.svd(essential_mat)

    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    if np.linalg.det(u @ w @ v) < 0:
        w = -w

    uwv = u @ w @ v
    uwtv = u @ w.T @ v
    u_bar = u[:, 2].reshape([-1, 1]) / abs(u[:, 2]).max()

    return [
        np.concatenate([uwv, u_bar], axis=1),
        np.concatenate([uwv, -u_bar], axis=1),
        np.concatenate([uwtv, u_bar], axis=1),
        np.concatenate([uwtv, -u_bar], axis=1),
    ]


def get_valid_triangulation(essential_mat, k2, p1, pts1, pts2):
    for m2 in m2s(essential_mat):
        p2 = k2 @ m2
        p3d, _ = triangulate(p1, pts1, p2, pts2)

        # return points if all points are in front of camera
        if all(z > 0 for z in p3d[:, 2]):
            return p3d

    raise ValueError('No valid camera matrix')


def reconstruct_using_calibration(pts1, pts2, fundamental_mat, k1, k2):
    """
    :param pts1: points in image 1
    :param pts2: matching points in image 2
    :param fundamental_mat: between camera 1 and 2
    :param k1: intrinsics camera 1
    :param k2: intrinsics camera 2
    :return:
    """
    # Convert fundamental to essential
    essential_mat = k2.T @ fundamental_mat @ k1

    m1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    p1 = k1 @ m1

    return get_valid_triangulation(essential_mat, k2, p1, pts1, pts2)
