import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import Tuple, Sequence
from numpy.linalg import svd, cholesky, inv
from scipy.linalg import null_space
from linear_algebra import cross_product_matrix, extend_to_3d
from features import Matches


class IAC:
    def __init__(self):
        # Constraints are sorted by lines:
        # w11, w12, w13,        w22, w23,      w33

        # Define symmetry constraints
        self._constraints = np.empty((0, 6), dtype=float)

    def add_scene_orthogonality_constraint(self, v1: npt.ArrayLike, v2: npt.ArrayLike) -> 'IAC':
        """
        Add constraint arising from scene orthogonality.

        :param v1: The vanishing point of the first orthogonal lines
        :param v2: The vanishing point of the second orthogonal lines
        :return: The IAC object
        """
        v1 = extend_to_3d(v1)
        v2 = extend_to_3d(v2)

        constraint = []
        for i in range(3):
            for j in range(3):
                constraint.append(v1[i] * v2[j])

        reduced_constraint = [
            constraint[0],                      # w11
            constraint[1] + constraint[3],      # w12
            constraint[2] + constraint[6],      # w13
            constraint[4],                      # w22
            constraint[5] + constraint[7],      # w23
            constraint[8]                       # w33
        ]
        self._constraints = np.vstack((self._constraints, np.array([reduced_constraint])))

        return self

    def add_square_pixels_constraints(self, ratio: float = 1.) -> 'IAC':
        """
        Add constraint arising from the ratio between the pixel width and height.

        :param ratio: The ratio of x scale to the y scale
        :return: The IAC object
        """
        self._constraints = np.vstack((
            self._constraints,
            (
                (0, 1, 0,              0, 0,    0),  # Zero skew w12 = w21 = 0
                ((ratio ** 2), 0, 0,   -1, 0,   0)   # Square pixels w11 = ratio^-2 * w22
            )))

        return self

    def solve(self) -> npt.ArrayLike:
        """
        Solve the constraints

        :return: The solved matrix of the IAC
        """
        _, _, vh = svd(self._constraints)
        solution = vh[-1]

        return np.array([
            [solution[0], solution[1], solution[2]],
            [solution[1], solution[3], solution[4]],
            [solution[2], solution[4], solution[5]]
        ])


def _generate_cameras(F: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Generate cameras for the given fundamental matrix, one of which is canonical.

    :param F: The fundamental matrix
    :return: A tuple with two cameras corresponding to the fundamental matrix
    """
    e2 = null_space(F.T, 1e-8).squeeze()

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((cross_product_matrix(e2) @ F, e2[:, None]))

    return P1, P2


class Reconstruction:
    def __init__(self, fundamental_matrix: npt.ArrayLike):
        self._f = fundamental_matrix
        self._p1, self._p2 = _generate_cameras(fundamental_matrix)
        self._intersection_matrix = np.hstack((np.vstack((self._p1, self._p2)), np.zeros((6, 2))))

    def reconstruct_point(self, x1: npt.ArrayLike, x2: npt.ArrayLike) -> npt.ArrayLike:
        """
        Reconstruct a 3D point from coordinates in both images.

        :param x1: The projection of the point in the first image
        :param x2: The projection of the point in the second image
        :return: The coordinates of the point
        """
        self._intersection_matrix[:3, 4] = extend_to_3d(x1)
        self._intersection_matrix[3:, 5] = extend_to_3d(x2)

        _, _, vh = svd(self._intersection_matrix)
        point = vh[-1, :4]
        return point[:3] / point[3]

    def reconstruct_matches(self, image1: npt.ArrayLike, matches: Matches) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        3D reconstruction of matches

        :param image1: The first image to read colors from
        :param matches: The matches to reconstruct
        :return: A 3D point cloud corresponding to the matches
        """
        reconstructed = []
        colors = []
        for np1, np2 in tqdm(zip(matches.first[matches[:, 0]].keypoints, matches.second[matches[:, 1]].keypoints)):
            reconstructed.append(self.reconstruct_point(np1, np2))
            coordinate = (np1 * image1.shape[:2]).astype(int)
            colors.append(image1[coordinate[0], coordinate[1], :])

        return np.stack(reconstructed), np.stack(colors)

    def metric_alignment(self, points1: npt.ArrayLike, points2: Sequence[npt.ArrayLike], image2):
        """
        Perform metric alignment based on parallel lines. Each parallel set should be of the shaep [L, 2, 2].
        The axes are set index, line index, point index, image axis index.

        :param image1: The first image.
        :param lines1: The first image parallel sets.
        :param image2: The second image.
        :param lines2: The second image parallel sets.
        """
        # Reconstruct vanishing points
        points3d = []
        for point1, point2 in zip(points1, points2):
            points3d.append(self.reconstruct_point(point1, point2))
        points3d = np.stack(points3d)

        # Find the plane at infinity (at `points3d` null-space)
        points3d = np.hstack((points3d, np.ones((3, 1))))
        _, _, vh = svd(points3d)
        inf_plane = vh[-1]

        # Construct the affine alignment homography
        homography = np.eye(4)
        homography[3, :] = inf_plane
        self.apply_homography(homography)

        # Find the IAC in the second image
        iac = IAC() \
            .add_scene_orthogonality_constraint(points2[0], points2[1]) \
            .add_scene_orthogonality_constraint(points2[0], points2[2]) \
            .add_scene_orthogonality_constraint(points2[1], points2[2]) \
            .add_square_pixels_constraints(image2.shape[0] / image2.shape[1])

        w = iac.solve()
        m = self._p2[:, :3]
        a = inv(cholesky(inv(m.T @ w @ m)))

        homography = np.eye(4)
        for i in range(3):
            homography[i, :3] = a[i]

        self.apply_homography(homography)

    def apply_homography(self, homography):
        """
        Applies an homography on the reconstruction

        :param homography: The homography to apply.
        """
        self._p1 = self._p1 @ homography
        self._p2 = self._p2 @ homography
