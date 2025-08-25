import numpy as np
import deepxde as dde

length_scale = 2
epsilon = 0.01
Nx = 128
Nt = 10000
Nt_target = 100
num_samples = 50
output_file = "allen_cahn_0.01.npz"

dx = 1 / Nx
dt = 1 / Nt
x = np.arange(0, 1, dx)
t_full = np.arange(0, 1, dt)
t = np.linspace(0, 1, Nt_target)

func_space = dde.data.GRF(length_scale=length_scale, kernel='ExpSineSquared')


def periodic_idx(i):
    return i % Nx


U_dataset = []
for sample_idx in range(num_samples):
    print(f"Generating sample {sample_idx + 1}/{num_samples}")
    func_feats = func_space.random(1)
    u_init = func_space.eval_batch(func_feats, x)[0].astype(np.float64)
    U = np.zeros((Nx, Nt))
    U[:, 0] = u_init

    # Allen-Cahn: u_t = epsilon^2 u_xx + u - u^3
    for n in range(Nt - 1):
        u_n = U[:, n].copy()
        u_np1 = u_n.copy()
        max_iter = 100
        tol = 1e-8
        for _ in range(max_iter):
            u_xx = (np.roll(u_np1, -1) - 2 * u_np1 + np.roll(u_np1, 1)) / dx ** 2
            u_new = u_n + dt * (epsilon ** 2 * u_xx + u_np1 - u_np1 ** 3)
            if np.max(np.abs(u_new - u_np1)) < tol:
                u_np1 = u_new
                break
            u_np1 = u_new
        U[:, n + 1] = u_np1

    indices = np.round(np.linspace(0, Nt - 1, Nt_target)).astype(int)
    U_downsampled = U[:, indices]
    U_dataset.append(U_downsampled)

U_dataset = np.array(U_dataset)
np.savez(output_file, x=x, t=t, U=U_dataset, epsilon=epsilon, length_scale=length_scale, Nx=Nx, Nt=Nt,
         Nt_target=Nt_target)
print(f"Dataset saved to {output_file}")
