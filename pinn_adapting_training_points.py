# -*- coding: utf-8 -*-

import os
import sys
from functools import partial

import torch
import time

from params import *
from src.adaptation import get_new_adapted_points, refine
from src.advection_dominated_loss_function import (
    compute_loss,
    f_inter_loss,
)
from src.mesh_1D import get_mesh
from src.pinn_core import PINN, f, train_model


# THE FIRST SET OF TRAINING POINTS IS AN EQUIDISTRIBUTED SET OF POINTS
x = torch.linspace(
    X_INI, X_FIN, steps=NUM_MAX_POINTS, requires_grad=True, device=DEVICE
).reshape(-1, 1)
base_x = torch.linspace(
    X_INI, X_FIN, steps=NUM_BASE_POINTS
)


pinn = PINN(LAYERS, NEURONS, pinning=False).to(DEVICE)

interior_loss_fn = partial(
    f_inter_loss, nn_approximator=pinn, epsilon=EPSILON
)


convergence_data = torch.empty(0)

# Keep all the points
point_data = []

n_iters = -1

start_time = time.time()
for i in range(MAX_ITERS):
    print(i)

    # UPDATE THE TRAINING POINTS FOR PINN
    loss_fn = partial(compute_loss, epsilon=EPSILON, x=x)

    # TRAIN PINN
    stage_convergence_data = train_model(
        pinn, loss_fn, DEVICE, learning_rate=LEARNING_RATE, max_epochs=NUMBER_EPOCHS
    )

    convergence_data = torch.cat((convergence_data, stage_convergence_data.cpu()))

    y = f(pinn, x).detach().cpu()
    plain_x = x.detach().clone().cpu()
    point_data.append( torch.stack((plain_x, y)).transpose(1, 0).reshape(-1, 2) )

    loss_fn = partial(f_inter_loss, epsilon=EPSILON, nn_approximator=pinn)
    x = refine(base_x, loss_fn, NUM_MAX_POINTS, TOL).to(DEVICE).requires_grad_(True)
    if x.numel() == NUM_BASE_POINTS:
        break

end_time = time.time()
exec_time = end_time - start_time

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

pinn = pinn.cpu()

torch.save(pinn, "results/data/pinn.pt")
torch.save(convergence_data.detach(), "results/data/convergence_data.pt")
torch.save(n_iters, "results/data/n_iters.pt")
torch.save(point_data, "results/data/point_data.pt")
torch.save(exec_time, "results/data/exec_time.pt")
