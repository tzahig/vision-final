import numpy as np
from numpy import cross
from numpy.linalg import inv
from numpy.linalg import svd

from linear_algebra import cross_product_matrix, extend_to_3d


def null_space(constraints):
    _, _, vh = svd(constraints)
    return vh[-1]


def omega_from_orthogonal_vanishing_points(vanishing_points, aspect_ratio):
    """
    :param vanishing_points: orthogonal vanishing points
    :param aspect_ratio: aspect of camera,  ratio of x scale to the y scale
    :return:
    """
    # Constraints are sorted by lines:
    # w11, w12, w13,        w22, w23,      w33

    # Define symmetry constraints
    constraints = np.empty((0, 6), dtype=float)

    def add_constraint(constraint):
        return np.vstack((constraints, np.array([constraint])))

    orthogonal_point_pairs = [
        (vanishing_points[0], vanishing_points[1]),
        (vanishing_points[0], vanishing_points[2]),
        (vanishing_points[1], vanishing_points[2]),
    ]

    for v1, v2 in orthogonal_point_pairs:
        v1 = extend_to_3d(v1)
        v2 = extend_to_3d(v2)

        # Add equation v1.T @ w @ v2 = 0
        orthogonality_constraint = [
            v1[0] * v2[0],  # coefficient of w11
            v1[0] * v2[1] + v1[1] * v2[0],  # coefficient of w12 = w21
            v1[0] * v2[2] + v1[2] * v2[0],  # coefficient of w13 = w31
            v1[1] * v2[1],  # coefficient of w22
            v1[1] * v2[2] + v1[2] * v2[1],  # coefficient of w23 = w32
            v1[2] * v2[2],  # coefficient of w33
        ]
        constraints = add_constraint(orthogonality_constraint)

    # Zero skew w12 = w21 = 0
    constraints = add_constraint((0, 1, 0, 0, 0, 0))

    # Square pixels w11 = aspect_ratio^-2 * w22:
    constraints = add_constraint((aspect_ratio ** 2, 0, 0, -1, 0, 0))

    solution = null_space(constraints)

    w = np.array(
        [
            [solution[0], solution[1], solution[2]],
            [solution[1], solution[3], solution[4]],
            [solution[2], solution[4], solution[5]],
        ]
    )
    return w
