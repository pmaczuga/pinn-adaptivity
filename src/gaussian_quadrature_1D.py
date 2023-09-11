# -*- coding: utf-8 -*-

import numpy as np
import torch


def d_transform(gp, gw, a, b):
    jac = 0.5 * (b - a)
    omega = jac * gw
    xq = a + jac * (1 + gp)
    return omega, xq


def gaussian_points_and_weights(order, local_device):
    if order == 1:
        gp = torch.tensor([[0.0]], device=local_device)
        gw = torch.tensor([[2.0]], device=local_device)
    elif order == 2:
        gp = torch.tensor(
            [[-0.57735027, 0.57735027]], device=local_device
        )  # gaussian points
        gw = torch.tensor([[1.0, 1.0]], device=local_device)
    elif order == 3:
        gp = torch.tensor(
            [[-0.77459667, 0.0, 0.77459667]], device=local_device
        )  # gaussian points
        gw = torch.tensor(
            [[0.55555556, 0.88888889, 0.55555556]], device=local_device
        )  # gaussian weights
    elif order == 4:
        gp = torch.tensor(
            [[-0.86113631, -0.33998104, 0.33998104, 0.86113631]], device=local_device
        )  # gaussian points
        gw = torch.tensor(
            [[0.34785485, 0.65214515, 0.65214515, 0.34785485]], device=local_device
        )  # gaussian weights
    elif order == 5:
        gp = torch.tensor(
            [[-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985]],
            device=local_device,
        )  # gaussian points
        gw = torch.tensor(
            [[0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]],
            device=local_device,
        )  # gaussian weights
    elif order == 6:
        gp = torch.tensor(
            [
                [
                    -0.93246951,
                    -0.66120939,
                    -0.23861919,
                    0.23861919,
                    0.66120939,
                    0.93246951,
                ]
            ],
            device=local_device,
        )
        gw = torch.tensor(
            [[0.17132449, 0.36076157, 0.46791393, 0.46791393, 0.36076157, 0.17132449]],
            device=local_device,
        )
    elif order == 7:
        gp = torch.tensor(
            [
                [
                    -0.94910791,
                    -0.74153119,
                    -0.40584515,
                    0.0,
                    0.40584515,
                    0.74153119,
                    0.94910791,
                ]
            ],
            device=local_device,
        )
        gw = torch.tensor(
            [
                [
                    0.12948497,
                    0.27970539,
                    0.38183005,
                    0.41795918,
                    0.38183005,
                    0.27970539,
                    0.12948497,
                ]
            ],
            device=local_device,
        )
    elif order == 8:
        gp = torch.tensor(
            [
                [
                    -0.96028986,
                    -0.79666648,
                    -0.52553241,
                    -0.18343464,
                    0.18343464,
                    0.52553241,
                    0.79666648,
                    0.96028986,
                ]
            ],
            device=local_device,
        )
        gw = torch.tensor(
            [
                [
                    0.10122854,
                    0.22238103,
                    0.31370665,
                    0.36268378,
                    0.36268378,
                    0.31370665,
                    0.22238103,
                    0.10122854,
                ]
            ],
            device=local_device,
        )
    else:
        ##################
        # Using numpy... #
        ##################
        gp, gw = np.polynomial.legendre.leggauss(order)
        gp = torch.tensor(gp, device=local_device, dtype=torch.float32).reshape(1, -1)
        gw = torch.tensor(gw, device=local_device, dtype=torch.float32).reshape(1, -1)

    return gp, gw


def get_gaussian_points_physical_space(xini, xfin, order):
    gp, gw = gaussian_points_and_weights(order, xini.device)
    return d_transform(gp, gw, xini, xfin)[1]
