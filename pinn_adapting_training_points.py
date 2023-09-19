# -*- coding: utf-8 -*-

import os
import sys
from functools import partial

import torch

from params import *
from src.adaptation import get_new_adapted_points
from src.advection_dominated_loss_function import (
    compute_loss,
    f_inter_loss,
)
from src.mesh_1D import get_mesh
from src.pinn_core import PINN, f, train_model


# THE FIRST SET OF TRAINING POINTS IS AN EQUIDISTRIBUTED SET OF POINTS
x = torch.linspace(
    X_INI, X_FIN, steps=NUM_POINTS + 2, requires_grad=True, device=DEVICE
)[1:-1]
x, _ = get_mesh(x, X_INI, X_FIN)


nn_approximator = PINN(LAYERS, NEURONS, pinning=False).to(DEVICE)

last_loss_values = torch.tensor(TOL * 10).repeat(3).cpu()

interior_loss_fn = partial(
    f_inter_loss, nn_approximator=nn_approximator, epsilon=EPSILON
)


number_total_evaluations = 0
convergence_data = torch.empty(0)

# Keep all the points
point_data = torch.empty((MAX_ITERS, NUM_POINTS + 2, 2))

n_iters = -10

for i in range(MAX_ITERS):
    if last_loss_values.mean() < TOL:
        n_iters = i
        break

    assert x.numel() == (NUM_POINTS + 2)
    print(i)

    # UPDATE THE TRAINING POINTS FOR PINN
    loss_fn = partial(compute_loss, epsilon=EPSILON, x=x)

    # TRAIN PINN
    stage_convergence_data = train_model(
        nn_approximator, loss_fn, DEVICE, learning_rate=LEARNING_RATE, max_epochs=NUMBER_EPOCHS
    )

    convergence_data = torch.cat((convergence_data, stage_convergence_data.cpu()))

    loss_value = stage_convergence_data[-1].reshape(-1)

    last_loss_values = torch.cat((last_loss_values[1:], loss_value.cpu()))

    number_total_evaluations = number_total_evaluations + NUMBER_EPOCHS * x.numel()

    y = f(nn_approximator, x).detach().cpu()
    plain_x = x.detach().clone().cpu()
    point_data[i, :] = torch.stack((plain_x, y)).transpose(1, 0).reshape(-1, 2)

    new_x = get_new_adapted_points(interior_loss_fn, x.reshape(-1), NUM_POINTS)
    x = torch.cat(
        (
            x[0],
            new_x,
            x[-1],
        )
    ).reshape(-1, 1)
    x = x.detach().clone().requires_grad_(True)


if n_iters == -10:
    sys.exit(f"The error tolerance has not been reached in {MAX_ITERS} iterations")

# Create result directory if it doesn't exist
try:
    os.makedirs("results/iterations")
except OSError as error:
    pass
try:
    os.makedirs("results/data")
except OSError as error:
    pass

nn_approximator = nn_approximator.cpu()

torch.save(nn_approximator, "results/data/pinn.pt")
torch.save(convergence_data.detach(), "results/data/convergence_data.pt")
torch.save(n_iters, "results/data/n_iters.pt")
torch.save(point_data, "results/data/point_data.pt")
