# -*- coding: utf-8 -*-

import numpy as np
import torch

from .pinn_core import PINN, dfdx, f


def exact_solution(x, eps):
    return 2 * (1 - torch.exp((x - 1) / eps)) / (1 - np.exp(-2 / eps)) + x - 1


def f_inter_loss(x, nn_approximator, epsilon):
    return (
        -epsilon * dfdx(nn_approximator, x, order=2)
        + dfdx(nn_approximator, x, order=1)
        - 1.0
    )


def compute_loss(
    nn_approximator: PINN, x: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Compute the full loss function as interior loss + boundary loss
    This custom loss function is fully defined with differentiable tensors therefore
    the .backward() method can be applied to it
    """

    # PDE residual
    interior_loss = f_inter_loss(x[1:-1], nn_approximator, epsilon)

    # u(1)=0
    boundary_xi = x[-1].reshape(-1, 1)  # last point = 1
    assert (boundary_xi == 1).all().item(), f"{boundary_xi}"
    boundary_loss_right = f(nn_approximator, boundary_xi)

    # u(-1)=0
    boundary_xf = x[0].reshape(-1, 1)  # first point = 0
    assert (boundary_xf == -1).all().item(), f"{boundary_xf}"
    boundary_loss_left = f(nn_approximator, boundary_xf)

    # obtain the final MSE loss by averaging each loss term and summing them up
    # final_loss = torch.cat((
    #     interior_loss,
    #     boundary_loss_left,
    #     boundary_loss_right)).pow(2).mean()

    final_loss = (
        interior_loss.pow(2).mean()
        + boundary_loss_left.pow(2).mean()
        + boundary_loss_right.pow(2).mean()
    )

    return final_loss
