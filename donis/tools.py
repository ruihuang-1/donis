import numpy as np
import torch
from pyDOE import lhs


def init_env(server_gpu: int):
    if not torch.cuda.is_available():
        return 0
    if torch.cuda.device_count() == 1:
        return 1
    if torch.cuda.device_count() > 2:
        torch.cuda.set_device(server_gpu)
        return 2


def mean_l2_relative_error(y_true, y_pred):
    return torch.norm(y_true - y_pred) / torch.norm(y_true)


def mean_l2_relative_error_on_batch(y_true, y_pred):
    if len(y_true.shape) != 2:
        raise RuntimeError('Input data must have exactly 2 dims.')
    return torch.mean(torch.norm(y_true - y_pred, dim=1) / torch.norm(y_true, dim=1))


def mean_square_error(y_true, y_pred):
    return torch.mean(torch.square(y_true - y_pred))


def format_dict(d, gap1='', gap2='_'):
    if d is None or len(d) == 0:
        return '--'
    text = ''
    for key, val in d.items():
        if type(val) is float or type(val) is np.float_:
            text += '{}{}{:.2e}{}'.format(key, gap1, val, gap2)
        else:
            text += '{}{}{}{}'.format(key, gap1, val, gap2)
    text = text[:-len(gap2)]
    return text


def interp_eval(interp, xs):
    """ does faster interpolation."""
    if not callable(interp):
        raise TypeError('interp is not callable')
    if len(xs.shape) != 2:
        raise RuntimeError('xs must be a 1 dim array')
    y = interp(xs)
    return np.array(y, dtype=np.float32).reshape(-1, 1)


def interps_eval(interps, xs):
    """ does faster interpolations. xs can be a 1d/2d numpy array, or a list of numpy array.
        returns a 2d numpy array: (N1*N2, 1)"""
    if type(interps) is not list and type(interps) is not np.ndarray:
        raise TypeError('interps must be a list/ndarray')

    if type(xs) is np.ndarray:
        if len(xs.shape) > 1 and xs.shape[-1] == 1:
            xs = xs.squeeze(axis=-1)
        if len(xs.shape) == 1:
            # fixed xs
            y = list(map(lambda interp: interp(xs), interps))  # faster than forloop
            return np.array(y, dtype=np.float32).reshape(-1, 1)
        elif len(xs.shape) == 2:
            # varied xs
            y = list(map(lambda i_interp: i_interp[1](xs[i_interp[0]]), enumerate(interps)))
            return np.array(y, dtype=np.float32).reshape(-1, 1)
        else:
            raise RuntimeError('xs has unexpected dims')

    elif type(xs) is list and type(xs[0]) is np.ndarray:
        # varied xs with varied lens
        # y = [interp(xs[i]) for i, interp in enumerate(interps)]
        y = list(map(lambda i_interp: i_interp[1](xs[i_interp[0]]), enumerate(interps)))
        return np.concatenate(y, dtype=np.float32).reshape(-1, 1)

    else:
        raise TypeError('xs must be a (list of) numpy array')


def lhs_sampling(ranges, m):
    """
    Generate M points in an N-dimensional space using LHS, scaled to the provided ranges.

    Parameters:
    - ranges: List of tuples [(min1, max1), (min2, max2), ..., (minN, maxN)] specifying the range for each dimension.
    - M: Total number of points to sample.

    Returns:
    - A numpy array of shape (M, N) where N is the number of dimensions.
    """
    # Number of dimensions
    n = len(ranges)

    # Generate LHS samples in the unit cube [0,1]^N
    lhs_points = lhs(n, samples=m)

    # Scale the points to the specified ranges
    for i in range(n):
        min_val, max_val = ranges[i]
        lhs_points[:, i] = lhs_points[:, i] * (max_val - min_val) + min_val

    return lhs_points.astype(np.float32)


def grid_uniform_sampling(ranges, total_points):
    """
    Generate uniformly distributed points in an N-dimensional space on a grid.

    Args:
        ranges (list of tuples): A list of N tuples, each specifying the (min, max) range for a dimension.
        total_points (int): Total number of points to sample.

    Returns:
        numpy.ndarray: A total_points x N array of sampled coordinates.
    """
    dimensions = len(ranges)

    # Compute the approximate number of points per dimension
    points_per_dim = int(np.round(total_points ** (1 / dimensions)))

    # Generate grid for each dimension
    grids = [np.linspace(low, high, points_per_dim) for low, high in ranges]

    # Create the cartesian product of the grids
    mesh = np.meshgrid(*grids, indexing='ij')
    grid_points = np.stack(mesh, axis=-1).reshape(-1, dimensions)

    # If the number of grid points exceeds the desired total_points, sample randomly
    if len(grid_points) > total_points:
        selected_indices = np.random.choice(len(grid_points), total_points, replace=False)
        grid_points = grid_points[selected_indices]

    return grid_points


def norm_feats(feats):
    """Project 1d functions to [0, 1] in batches."""
    if len(feats.shape) != 2:
        raise RuntimeError('This function deals with 2d inputs.')
    return (feats - feats.min(axis=1)[:, None]) / (feats.max(axis=1)[:, None] - feats.min(axis=1)[:, None])
