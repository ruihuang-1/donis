import argparse
import os
import sys
from tqdm import tqdm
from donis.tools import *

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import deepxde as dde

x_range = (0, 1)
t_range = (0, 1)

def solve_adr(xmin, xmax, tmin, tmax, k_fun, v_fun, g_fun, dg_fun, f_fun, u0_fun, Nx, Nt):
    """
    Solve the 1D Advection-Diffusion-Reaction (ADR) PDE:
        u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x,t),
    with homogeneous Dirichlet boundary conditions (u=0 at boundaries).

    Notes
    -----
    This implementation is adapted from an example in DeepXDE
    (LGPL v2.1: https://github.com/lululxvi/deepxde/blob/master/examples/operator/ADR_solver.py).
    """
    # --- grids ---
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    dx2 = dx * dx

    # --- finite difference operators ---
    # first-order centered difference (off-diagonal)
    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    # second-order centered difference (tridiagonal Laplacian)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    # interior identity (for excluding boundaries)
    I_interior = np.eye(Nx - 2)

    # --- coefficients ---
    kx = k_fun(x)
    vx = v_fun(x)

    # diffusion operator
    M_diff = -np.diag(D1 @ kx) @ D1 - 4.0 * np.diag(kx) @ D2
    # restrict to interior nodes
    M_core = M_diff[1:-1, 1:-1]

    # advection operator
    adv_matrix = 2 * dx * np.diag(vx[1:-1]) @ D1[1:-1, 1:-1]
    adv_matrix += 2 * dx * np.diag(vx[2:] - vx[:-2])

    # assemble LHS/RHS constant parts
    lhs_base = (8 * dx2 / dt) * I_interior + M_core + adv_matrix
    rhs_base = (8 * dx2 / dt) * I_interior - M_core - adv_matrix

    # source term sampled on grid
    f_vals = f_fun(x[:, None], t)

    # --- solution array ---
    u = np.zeros((Nx, Nt))
    u[:, 0] = u0_fun(x)

    # --- time stepping (implicit scheme with Newton linearization) ---
    for n in range(Nt - 1):
        u_curr = u[1:-1, n]
        g_vals = g_fun(u_curr)
        dg_vals = dg_fun(u_curr)

        nonlinear_term = np.diag(4.0 * dx2 * dg_vals)
        A = lhs_base - nonlinear_term
        rhs_vec = rhs_base @ u_curr + 8 * dx2 * (
            0.5 * f_vals[1:-1, n] + 0.5 * f_vals[1:-1, n + 1] + g_vals
        )

        u[1:-1, n + 1] = np.linalg.solve(A, rhs_vec)

    return x, t, u



def gen_data(size, is_cartesian: bool):
    func_space = dde.data.GRF(length_scale=0.2)
    test_feats = func_space.random(size)
    xs = np.linspace(0, 1, num=100)[:, None]
    u_true = []
    for v in tqdm(func_space.eval_batch(test_feats, xs)):
        x, t, u = solve_adr(
            xmin=x_range[0],
            xmax=x_range[1],
            tmin=t_range[0],
            tmax=t_range[1],
            k_fun=lambda x: 0.01 * np.ones_like(x),
            v_fun=lambda x: np.zeros_like(x),
            g_fun=lambda u: 0.01 * u ** 2,
            dg_fun=lambda u: 0.02 * u,
            f_fun=lambda x, t: np.tile(v[:, None], (1, len(t))),  # source term
            u0_fun=lambda x: np.zeros_like(x),
            Nx=100,
            Nt=100,
        )
        u_true.append(u.T)

    u_true = np.array(u_true)
    target = u_true
    v_branch = func_space.eval_batch(test_feats, np.linspace(0, 1, num=50)[:, None])
    xv, tv = np.meshgrid(np.linspace(x_range[0], x_range[1], num=100), np.linspace(t_range[0], t_range[1], num=100))
    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T

    if not is_cartesian:
        v_branch = v_branch.repeat(repeats=len(x_trunk), axis=0).reshape(-1, 50)
        x_trunk = np.tile(x_trunk, (size, 1, 1)).reshape(-1, 2)

    np.savez('ndr.npz', input_v=v_branch, input_x=x_trunk, target=target, feats=test_feats)


if __name__ == '__main__':
    dde.config.set_random_seed(101)
    parser = argparse.ArgumentParser(description='diff-react test set generation')
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--cartesian', type=bool, default=False)
    args = parser.parse_args()
    gen_data(size=args.size, is_cartesian=args.cartesian)
