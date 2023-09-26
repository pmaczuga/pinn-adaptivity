# -*- coding: utf-8 -*-

from typing import Callable

import torch
from scipy.integrate import quad

from src.gaussian_quadrature_1D import get_gaussian_points_physical_space
from src.random_points import get_id_points

Function = Callable[[torch.Tensor], torch.Tensor]


def get_new_adapted_points(
    f: Function, input_points: torch.Tensor, n_points_out: int
) -> torch.Tensor:
    x = input_points
    t1 = torch.cat(
        (
            torch.cat(
                tuple(
                    map(
                        lambda gp: get_gaussian_points_physical_space(
                            gp[0], gp[1], 7
                        ).reshape(-1),
                        zip(x[:-1], x[1:]),
                    )
                )
            ),
            x[1:-1],
        )
    ).unique()
    t1 = t1[(t1 != x[0]) & (t1 != x[-1])]
    # t1 = t1[t1 != 1.0]
    _, idx = f(t1.reshape(-1, 1)).reshape(-1).sort()
    t2 = t1[idx][:n_points_out]
    return t2.sort()[0]


def np_to_torch(v):
    return torch.tensor(v, dtype=torch.float32).cpu().reshape(-1, 1)


def refine(base_x: torch.Tensor, loss_fun: Callable, num_max_points: int, tol: float):
    def loss_np(x):
        return loss_fun(x=np_to_torch(x)).reshape(-1)

    x = base_x.detach().clone().requires_grad_(True)
    n_points = x.numel()
    refined = True

    while n_points < num_max_points and refined:
        refined = False
        new_points = []
        for x1, x2 in zip(x[:-1], x[1:]):
            int_x = (
                torch.linspace(x1.item(), x2.item(), 20)
                .requires_grad_(True)
                .reshape(-1, 1)
                .to(x.device)
            )
            int_y = loss_fun(x=int_x) ** 2
            el_loss = torch.trapezoid(int_y, int_x, dim=0) / (x2 - x1)
            # el_loss = quad(loss_np, x1.item(), x2.item())[0] / (x2 - x1)
            if el_loss > tol:
                refined = True
                new_points.append((x1 + x2) / 2.0)
        x = torch.cat((x, torch.tensor(new_points, device=x.device))).sort()[0]
        n_points = x.numel()
    return x.reshape(-1, 1).detach().clone().requires_grad_(True)

def exit_criterion_no_adaptation(base_x: torch.Tensor, loss_fun: Callable, tol: float):
    def loss_np(x):
        return loss_fun(x=np_to_torch(x)).reshape(-1)

    x = base_x.detach().clone().requires_grad_(True)

    for x1, x2 in zip(x[:-1], x[1:]):
        int_x = (
            torch.linspace(x1.item(), x2.item(), 20)
            .requires_grad_(True)
            .reshape(-1, 1)
            .to(x.device)
        )
        int_y = loss_fun(x=int_x) ** 2
        el_loss = torch.trapezoid(int_y, int_x, dim=0) / (x2 - x1)
        # el_loss = quad(loss_np, x1.item(), x2.item())[0] / (x2 - x1)
        if el_loss > tol:
            return False

    return True
