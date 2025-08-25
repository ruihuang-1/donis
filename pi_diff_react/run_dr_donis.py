import os
import sys

import time
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from tqdm import trange

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import deepxde as dde
from donis.tools import *
from donis.sampling import func_sampling, pts_sampling_v

GPU = 0

params = {
    'name': 'NDR_IS',
    "epoch": 30000,
    "lr": 5e-4,
    "seed": 2067,

    "enable_isf": True,
    "enable_isc": True,

    "fixed_grid": False,
    "grid_size": 50,  # determines grid size of isc
    "interp_type": "cubic",  # options: cubic, idw, wind, krig
    "isc_eval_num": 200,  # seed num. actual num may change after random sampling
    "do_end_isc": False,
    "isc_end_step": 5000,  # determined by do_end_isc

    "isf_eval_num": 50,  # actual num may change after random sampling
    "do_reweighting": True,

    "enable_grad_clip": False,
    "clip_norm": 1.0,

    "num_func": 1000,
    "funcs_batch": 50,
    "num_test_func": 50,
    "num_val": 50,
    "val_path": "ndr.npz",
    "x_range": (0., 1.),
    "t_range": (0., 1.),
    "num_ic": 500,
    "num_bc": 500,
    "num_dom": 2000,

    "trunk": [2, 128, 128, 128, 128, 128],
    "branch": [50, 128, 128, 128, 128, 128],

    'val_every': 10,
    'save_every': 5000,
}

# Reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
dde.config.set_random_seed(params['seed'])

# Logger
start_time = time.strftime("%m%d_%H%M%S", time.localtime())
file_name = '{}#{}'.format(start_time, params['name'])
path = 'runs/' + file_name
writer = SummaryWriter(path)
print(start_time, params['name'])

# GPU Settings
ON_CUDA = torch.cuda.is_available()
ENV = init_env(server_gpu=GPU)


def main():
    # NN
    net = dde.nn.DeepONet(
        layer_sizes_branch=params['branch'],  # branch
        layer_sizes_trunk=params['trunk'],  # trunks
        activation="tanh",
        kernel_initializer="Glorot normal",
    )
    # Function Space
    func_space = dde.data.GRF(length_scale=0.2)
    feats = func_space.random(params['num_func'] + params['num_test_func'])
    train_feats = feats[:params['num_func']]
    test_feats = feats[params['num_func']:]
    func_pnts = np.linspace(params['x_range'][0], params['x_range'][1], params['branch'][0])
    train_vals = func_space.eval_batch(train_feats, func_pnts)
    test_vals = func_space.eval_batch(test_feats, func_pnts)
    train_interps = np.array([
        interp1d(np.ravel(func_space.x), feature, kind=func_space.interp, copy=False, assume_sorted=True)
        for feature in train_feats])
    test_interps = np.array([
        interp1d(np.ravel(func_space.x), feature, kind=func_space.interp, copy=False, assume_sorted=True)
        for feature in test_feats])
    train_x, train_x_dom, train_x_ic, train_x_bc = gen_col_pts(params['x_range'], params['t_range'], params['num_dom'],
                                                               params['num_ic'], params['num_bc'], params['num_func'])
    test_x, test_x_dom, test_x_ic, test_x_bc = gen_col_pts(params['x_range'], params['t_range'], params['num_dom'],
                                                           params['num_ic'], params['num_bc'], params['num_test_func'])

    # Importance Sampling
    if params['enable_isf']:
        isf_eval_x = grid_uniform_sampling([params['x_range'], params['t_range']],
                                           params['isf_eval_num']).astype(
            np.float32)  # fixed
        isf_eval_num = len(isf_eval_x)
        print('Actual isf_eval_num: {}'.format(isf_eval_num))
        isf_eval_vx = interps_eval(train_interps, isf_eval_x[:, 0])

    if params['enable_isc']:
        grid_x, grid_y = np.mgrid[
                         params['x_range'][0]:params['x_range'][1]:params['grid_size'] * 1j,
                         params['t_range'][0]:params['t_range'][1]:params['grid_size'] * 1j]
        grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
        isc_eval_xs = np.tile(
            lhs_sampling((params['x_range'], params['t_range']), params['isc_eval_num']),
            (params['num_func'], 1, 1))
        corners = np.array(list(product(params['x_range'], params['t_range'])), dtype=np.float32)
        corners = np.tile(corners, (params['num_func'], 1, 1))
        isc_eval_xs = np.concatenate((isc_eval_xs, corners), axis=1)

    # Data Val
    data = np.load(params['val_path'], allow_pickle=True)
    val_x = data['input_x'].astype(np.float32)
    val_v = data['input_v'].astype(np.float32)
    val_u = data['target'].astype(np.float32).reshape(params['num_val'], -1)
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    # Train
    pbar = trange(params['epoch'])
    mses, l2res = [], []
    best_model = net.state_dict()
    best_val = float('inf')
    isc_end_step = params['isc_end_step'] if params['do_end_isc'] else params['epoch']
    for epoch in pbar:
        if params['enable_isf']:
            func_idx, isf_weights = func_sampling(net, pde, params['funcs_batch'], isf_eval_x, train_vals,
                                                  vx=isf_eval_vx)  # x v vx
        else:
            func_idx, isf_weights = np.random.choice(range(params['num_func']), params['funcs_batch']), 1.

        col_pts = train_x[func_idx].copy()
        if params['enable_isc'] and epoch < isc_end_step:
            isc_eval_vx = interps_eval(train_interps[func_idx], isc_eval_xs[func_idx, :, 0]).reshape(-1, 1)
            new_grid, col_dom, isc_weights = pts_sampling_v(net, pde, params['num_dom'], grid, func_idx, isc_eval_xs,
                                                            train_vals, vx=isc_eval_vx, interp=params['interp_type'],
                                                            grid_size=params['grid_size'])
            col_pts[:, :params['num_dom']] = col_dom
            if not params['fixed_grid']:
                isc_eval_xs = new_grid  # fixed corners when updating evals
        else:
            isc_weights = 1.
        weights = torch.tensor(isf_weights * isc_weights)

        # TRAIN
        train_vx = interps_eval(train_interps[func_idx], col_pts[:, :, 0])
        train_pde, train_ic, train_bc = train_step(net, optimizer, train_vals[func_idx], col_pts, weights, train_vx)
        # TEST
        test_vx = interps_eval(test_interps, test_x[:, :, 0])
        test_pde, test_ic, test_bc = test_step(net, test_vals, test_x, test_vx)
        # EXTRA_TEST
        if epoch % 100 == 0:  # evaluate train loss with uniform grid
            extra_vx = interps_eval(train_interps, train_x[:, :, 0]).reshape(len(train_interps), -1)
            extra_pde, extra_ic, extra_bc = extra_test_step(net, train_vals, train_x, extra_vx, params['num_func'],
                                                            params['funcs_batch'])
            writer.add_scalar('3 eTrain Loss/0 ALL', extra_pde + extra_ic + extra_bc, epoch)
            writer.add_scalar('3 eTrain Loss/1 PDE', extra_pde, epoch)
            writer.add_scalar('3 eTrain Loss/2 IC', extra_ic, epoch)
            writer.add_scalar('3 eTrain Loss/3 BC', extra_bc, epoch)

        # VAL
        if (epoch + 1) % params['val_every'] == 0 or epoch == 0:
            mse, l2re = val_step(net, val_u, val_x, val_v)
            if (epoch + 1) % params['save_every'] == 0:
                os.makedirs('saves/{}#{}'.format(start_time, params['name']), exist_ok=True)
                torch.save(net.state_dict(),
                           'saves/{}/{:06d}#[{:.2e},{:.2e}]#{}.pth'.format(file_name, epoch, mse, l2re, file_name))
                print('\n"{}" saved at {}, [{:.2e}, {:.2e}]'.format(file_name, epoch, mse, l2re))
            if mse < best_val:
                best_val = mse
                best_model = net.state_dict()
            mses.append(mse)
            l2res.append(l2re)
            writer.add_scalar('0 Val/0 MSE', mse, epoch)
            writer.add_scalar('0 Val/1 L2RE', l2re, epoch)

        writer.add_scalar('1 Train Loss/0 ALL', train_pde + train_ic + train_bc, epoch)
        writer.add_scalar('1 Train Loss/1 PDE', train_pde, epoch)
        writer.add_scalar('1 Train Loss/2 IC', train_ic, epoch)
        writer.add_scalar('1 Train Loss/3 BC', train_bc, epoch)
        writer.add_scalar('2 Test Loss/0 ALL', test_pde + test_ic + test_bc, epoch)
        writer.add_scalar('2 Test Loss/1 PDE', test_pde, epoch)
        writer.add_scalar('2 Test Loss/2 IC', test_ic, epoch)
        writer.add_scalar('2 Test Loss/3 BC', test_bc, epoch)

        stats = {'train': '[pde {:.2e} ic {:.2e} bc {:.2e}]'.format(train_pde, train_ic, train_bc),
                 'test': '[pde {:.2e} ic {:.2e} bc {:.2e}]'.format(test_pde, test_ic, test_bc),
                 'val': '[mse {:.2e} l2re {:.2e}]'.format(mses[-1], l2res[-1])}
        pbar.set_postfix(stats)

    # Complete
    mses = np.array(mses)
    l2res = np.array(l2res)
    best_step = np.argmin(mses)
    writer.add_text('Params', str(params))
    writer.add_text('Results', str({'mse': mses[best_step], 'l2re': l2res[best_step]}))
    writer.close()

    m_info = '[{:.2e},{:.2e}]'.format(mses[best_step], l2res[best_step])
    print('{} {} done.\nBest step at {}: {}'.format(start_time, params['name'], best_step, m_info))
    os.rename(path, 'runs/@{}#{}#{}'.format(start_time, params['name'], m_info))

    os.makedirs('saves/{}#{}'.format(start_time, params['name']), exist_ok=True)
    torch.save(best_model,
               'saves/{}/BEST#[{:.2e}]#{}.pth'.format(file_name, best_val, file_name))


def pde(x, y, v):
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return dy_t - 0.01 * dy_xx + 0.01 * y ** 2 - v


def gen_col_pts(x_range, t_range, num_dom, num_ic, num_bc, num_function):
    x_dom = lhs_sampling((x_range, t_range), num_dom).astype(np.float32)
    x_ic = np.column_stack((np.linspace(0, 1, num_ic), np.zeros(num_ic))).astype(np.float32)
    x_bc = np.vstack(
        (np.column_stack((np.zeros(num_bc // 2), np.linspace(0, 1, num_bc // 2))),
         np.column_stack(
             (np.ones(num_bc // 2), np.linspace(0, 1, num_bc // 2))))).astype(np.float32)
    x = np.tile(np.vstack((x_dom, x_ic, x_bc)), (num_function, 1, 1))
    return x, x_dom, x_ic, x_bc


def train_step(net, optimizer, train_vals, col_pts, weights, aux):
    net.train()
    optimizer.zero_grad()

    loss_pde, loss_ic, loss_bc = get_losses(net, train_vals, col_pts, weights, aux, params['do_reweighting'])
    loss = loss_bc + loss_ic + loss_pde
    loss.backward()

    # Gradient clipping
    if params['enable_grad_clip']:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=params['clip_norm'])  # 裁剪梯度的最大范数

    optimizer.step()

    train_pde, train_ic, train_bc = loss_pde.item(), loss_ic.item(), loss_bc.item()
    dde.grad.clear()

    return train_pde, train_ic, train_bc


def test_step(net, test_vals, test_x, aux):
    net.eval()
    loss_pde, loss_ic, loss_bc = get_losses(net, test_vals, test_x, None, aux, False, True)
    test_pde, test_ic, test_bc = loss_pde.item(), loss_ic.item(), loss_bc.item()
    dde.grad.clear()
    return test_pde, test_ic, test_bc


def extra_test_step(net, test_vals, test_x, aux, num_all, num_batch):
    if num_all % num_batch != 0:
        raise RuntimeError('Function Batch Invalid.')
    pdes = []
    ics = []
    bcs = []
    for i in range(num_all // num_batch):
        pde_i, ic_i, bc_i = test_step(net, test_vals[i:i + num_batch], test_x[i:i + num_batch],
                                      aux[i:i + num_batch])
        pdes.append(pde_i)
        ics.append(ic_i)
        bcs.append(bc_i)
    return sum(pdes) / len(pdes), sum(ics) / len(ics), sum(bcs) / len(bcs)


def val_step(net, val_u, val_x, val_v):
    net.eval()
    target = torch.tensor(val_u)
    x = torch.tensor(val_x.reshape(-1, 2), requires_grad=False)
    v = torch.tensor(val_v.reshape(-1, params['branch'][0]), requires_grad=False)
    output = net.forward((v, x)).detach().reshape(params['num_val'], -1)
    mse = mean_square_error(target, output)
    l2re = mean_l2_relative_error(target, output)
    return mse.item(), l2re.item()


def get_losses(net, func_vals, col_pts, weights, aux, reweighting, is_test=False):
    funcs_batch = params['funcs_batch']
    num_dom = params['num_dom']
    num_ic = params['num_ic']
    num_bc = params['num_bc']

    # Forward
    x = col_pts  # B*P*d1
    v = np.tile(func_vals[:, None, :], (1, x.shape[1], 1))  # B*P*d2
    x = torch.tensor(x.reshape(-1, params['trunk'][0]), requires_grad=True)
    v = torch.tensor(v.reshape(-1, params['branch'][0]), requires_grad=False)
    vx = torch.tensor(aux).reshape(-1, 1)
    output = net.forward((v, x))

    # pde loss
    error_pdes = pde(x, output, vx)
    if not isinstance(error_pdes, (list, tuple)):
        error_pdes = [error_pdes]
    error_pdes = torch.stack(error_pdes, dim=0)
    # MSE with Re-weighting
    loss_pde = torch.sum(torch.square(error_pdes), dim=0).reshape(funcs_batch, -1)  # Nf * Nc
    if reweighting:
        loss_pde[:, :num_dom] *= weights  # re-weighting (optional)
    loss_pde = torch.mean(loss_pde)

    # ic loss
    # zero ic
    output_ic = output.reshape(funcs_batch, -1)[:, num_dom:num_dom + num_ic]
    target_ic = torch.zeros_like(output_ic)
    loss_ic = mean_square_error(output_ic, target_ic)
    # bc loss
    # zero bc
    output_bc = output.reshape(funcs_batch, -1)[:, -num_bc:]
    target_bc = torch.zeros_like(output_bc)
    loss_bc = mean_square_error(output_bc, target_bc)

    return loss_pde, loss_ic, loss_bc


if __name__ == '__main__':
    main()
