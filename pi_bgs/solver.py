import numpy as np
import deepxde as dde

length_scale = 0.5
nu = 0.01
Nx = 128
Nt = 10000
Nt_target = 100
num_samples = 50
output_file = "burgers0.01.npz"

dx = 1 / Nx
dt = 1 / Nt
x = np.arange(0, 1, dx)
t_full = np.arange(0, 1, dt)
t = np.linspace(0, 1, Nt_target)

func_space = dde.data.GRF(length_scale=length_scale)


def Leftbdry(t):
    return 0


def Rightbdry(t):
    return 0


U_dataset = []
for sample_idx in range(num_samples):
    print(f"Generating sample {sample_idx + 1}/{num_samples}")

    func_feats = func_space.random(1)
    u = func_space.eval_batch(func_feats, x)[0].astype(np.float64)

    U = np.zeros((Nx, Nt))
    U[:, 0] = u

    for n in range(Nt - 1):
        u_n = U[:, n]
        u_np1 = u_n
        max_iter = 100
        tol = 1e-6

        for iter in range(max_iter):
            u_x = np.zeros(Nx)
            u_xx = np.zeros(Nx)

            u_x[1:-1] = (u_np1[2:] - u_np1[:-2]) / (2 * dx)
            u_xx[1:-1] = (u_np1[2:] - 2 * u_np1[1:-1] + u_np1[:-2]) / (dx ** 2)

            u_x[0] = (u_np1[1] - u_np1[0]) / dx
            u_x[-1] = (u_np1[-1] - u_np1[-2]) / dx
            u_xx[0] = (u_np1[1] - 2 * u_np1[0] + Leftbdry(t_full[n + 1])) / (dx ** 2)
            u_xx[-1] = (Rightbdry(t_full[n + 1]) - 2 * u_np1[-1] + u_np1[-2]) / (dx ** 2)

            uu_x = u_np1 * u_x
            u_np1_new = u_n - dt * (uu_x - nu * u_xx)

            if np.linalg.norm(u_np1_new - u_np1, np.inf) < tol:
                break
            u_np1 = u_np1_new

        U[:, n + 1] = u_np1
        U[0, n + 1] = Leftbdry(t_full[n + 1])
        U[-1, n + 1] = Rightbdry(t_full[n + 1])

    indices = np.round(np.linspace(0, Nt - 1, Nt_target)).astype(int)
    U_downsampled = U[:, indices]
    U_dataset.append(U_downsampled)

U_dataset = np.array(U_dataset)
np.savez(output_file, x=x, t=t, U=U_dataset, epsilon=nu, length_scale=length_scale, Nx=Nx, Nt=Nt,
         Nt_target=Nt_target)

print(f"Dataset saved to {output_file}")
