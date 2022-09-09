import numpy as np
import numpy.typing as npt


def cross_product_matrix(vec: npt.ArrayLike):
    """
    Returns the cross product matrix of a vector
    :param vec: The vector to construct the cross product matrix
    :return: The cross product matrix of the vector
    """
    return np.cross(vec, np.identity(vec.shape[0]) * -1)


def extend_to_3d(mat: npt.ArrayLike):
    """
    Extend a matrix of (H, 2) shape to (H, 3) by adding a ones column
    :param mat: The matrix to extend
    :return: The extended matrix
    """
    assert len(mat.shape) in (1, 2), "mat isn't in a valid shape"
    if len(mat.shape) == 1:
        assert mat.shape[0] == 2, "mat doesn't represent a 2D non-homogenous point"
        return np.concatenate((mat, [1]))
    else:
        assert mat.shape[1] == 2, "mat doesn't represent a 2D non-homogenous point matrix"
        return np.hstack((mat, np.ones((mat.shape[0], 1))))


def distance_of_point_from_line(p: npt.ArrayLike, l: npt.ArrayLike):
    """
    Finds the distance of a point from a line

    :param p: 3D vector of the point in homogenous coordinates
    :param l: 3D vector of the line in homogenous coordinates
    :return: The distance of the point from the line
    """
    denominator = p[2] * np.sqrt(l[0] ** 2 + l[1] ** 2)
    if denominator == 0:
        return 0 if np.isclose(np.dot(p, l), 0) else np.inf
    else:
        return np.abs(np.dot(p, l) / denominator)
