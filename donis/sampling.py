from scipy import ndimage
from scipy.interpolate import griddata

import deepxde as dde
from donis.tools import *
import pyinterp


def func_sampling(net, pde, bs, x, v, vx=None):
    num_x, dim_x = x.shape
    num_v, dim_v = v.shape
    x = torch.tensor(np.tile(x[None, :, :], (num_v, 1, 1)).reshape(-1, dim_x), requires_grad=True)
    v = torch.tensor(np.tile(v[:, None, :], (1, num_x, 1)).reshape(-1, dim_v))
    vx = torch.tensor(vx) if vx is not None else None

    # evaluate
    net.train()
    u = net.forward((v, x))
    errors = pde(x, u, vx)
    if not isinstance(errors, (list, tuple)):
        errors = [errors]
    errors = torch.stack(errors, dim=0).detach().cpu().numpy()
    error = np.sum(np.abs(errors), axis=0).reshape(num_v, -1)  # Nf * Nc
    scores = np.sum(error, axis=1)  # Nf
    dde.grad.clear()

    # sample
    probs = scores / scores.sum()
    indices = np.random.choice(num_v, bs, replace=False, p=probs)
    weights = 1 / (probs[indices] * num_v)
    return indices, weights[:, None]  # additional dim for broadcasting


def pts_sampling_v(net, pde, bs, grid, func_idx, xs, v, vx=None, grid_size=50, decay=0, interp='cubic'):
    x = xs[func_idx]
    v = v[func_idx]
    num_x, dim_x = x.shape[1:]
    num_v, dim_v = v.shape
    x = torch.tensor(x.reshape(-1, dim_x), requires_grad=True)
    v = torch.tensor(np.tile(v[:, None, :], (1, num_x, 1)).reshape(-1, dim_v))
    vx = torch.tensor(vx) if vx is not None else None

    # evaluate
    net.train()
    u = net.forward((v, x))
    errors = pde(x, u, vx)
    if not isinstance(errors, (list, tuple)):
        errors = [errors]
    errors = torch.stack(errors, dim=0).detach().cpu().numpy()  # N_pde * --
    error = np.sum(np.abs(errors), axis=0).reshape(num_v, -1)  # Nf * Nc
    dde.grad.clear()

    isc_weights = []
    coords = []

    new_xs = xs.copy()

    uniform_prob = np.full(len(grid), 1.0 / len(grid)).astype(np.float32)

    for i, fi in enumerate(func_idx):
        # interp
        scores_i = error[i]
        isc_eval_xs_i = xs[fi]

        # interp
        # cubic
        if interp == 'cubic':
            scores_interp = griddata(isc_eval_xs_i, scores_i, grid, method='cubic')
        else:
            mesh = pyinterp.RTree()
            mesh.packing(isc_eval_xs_i, scores_i)
            if interp == 'idw':
                scores_interp, neighbors = mesh.inverse_distance_weighting(
                    grid,
                    within=False,
                    k=11,
                    num_threads=0)
            elif interp == 'wind':
                scores_interp, neighbors = mesh.window_function(
                    grid,
                    within=False,
                    k=11,
                    wf='parzen',
                    num_threads=0)
            elif interp == 'krig':
                scores_interp, neighbors = mesh.universal_kriging(
                    grid,
                    within=False,
                    k=11,
                    covariance='matern_12',
                    alpha=100_000,
                    num_threads=0)
            else:
                raise RuntimeError('Wrong interp type.')

        scores_interp_abs = np.abs(scores_interp)

        # score_process
        if np.min(scores_interp_abs) < 5e-4:
            scores_interp_abs += 5e-4

        # sampling
        probs = scores_interp_abs / scores_interp_abs.sum()
        mixed_probs = (1 - decay) * probs + decay * uniform_prob  # uniform sampling when 1.0 decay
        probs = mixed_probs / mixed_probs.sum()

        indices = np.random.choice(len(grid), bs, replace=False, p=probs)
        coords_i = grid[indices]
        weights_i = 1 / (probs[indices] * len(grid))
        coords.append(coords_i)
        isc_weights.append(weights_i)

        # update seeds
        scores_interp = scores_interp.reshape(grid_size, grid_size)
        sobel_x = ndimage.sobel(scores_interp, axis=0)
        sobel_y = ndimage.sobel(scores_interp, axis=1)
        grads = np.hypot(sobel_x, sobel_y).flatten()
        probs_grads = grads / grads.sum()
        indices_grads = np.random.choice(len(grid), num_x - 4, replace=False,
                                         p=probs_grads)
        new_xs[fi][:-4] = grid[indices_grads]  # corner points stays the same

    return new_xs, np.array(coords, dtype=np.float32), np.array(isc_weights, dtype=np.float32)

