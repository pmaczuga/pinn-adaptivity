# -*- coding: utf-8 -*-

import sys
import os
from functools import partial

import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from src.adaptation import get_new_adapted_points
from src.advection_dominated_loss_function import compute_loss, exact_solution, f_inter_loss
from src.mesh_1D import get_mesh
from src.pinn_core import PINN, f, train_model

plt.rcParams["figure.dpi"] = 150
rc("animation", html="html5")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

X_INI = -1.0
LENGTH = 2.0
X_FIN = X_INI + LENGTH

NUM_POINTS = 200

EPSILON = 0.5  # <-HERE W CHANGE THE epsilon (=0.1 is working, =0.01 is not working)

MAX_ITERS = 1000
NUMBER_EPOCHS = 200

# Tolerance
TOL = 1e-5

# THE FIRST SET OF TRAINING POINTS IS AN EQUIDISTRIBUTED SET OF POINTS
x = torch.linspace(
    X_INI, X_FIN, steps=NUM_POINTS + 2, requires_grad=True, device=DEVICE
)[1:-1]
x, _ = get_mesh(x, X_INI, X_FIN)


nn_approximator = PINN(3, 15, pinning=False).to(DEVICE)

last_loss_values = torch.tensor(TOL*10).repeat(3).cpu()

interior_loss_fn = partial(
    f_inter_loss, nn_approximator=nn_approximator, epsilon=EPSILON
)


number_total_evaluations = 0
convergence_data = torch.empty(0)

# Keep all the points
point_data = torch.empty((MAX_ITERS, NUM_POINTS + 2, 2), device=DEVICE)

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
        nn_approximator, loss_fn, DEVICE, learning_rate=0.005, max_epochs=NUMBER_EPOCHS
    )

    convergence_data = torch.cat((convergence_data, stage_convergence_data.cpu()))

    loss_value = stage_convergence_data[-1].reshape(-1)

    last_loss_values = torch.cat((last_loss_values[1:], loss_value.cpu()))

    number_total_evaluations = number_total_evaluations + NUMBER_EPOCHS * x.numel()

    y = f(nn_approximator, x)
    point_data[i, :] = torch.stack((x, y)).transpose(1, 0).reshape(-1, 2)

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

for i, p in enumerate(point_data[:n_iters, :]):
    fig, ax = plt.subplots()
    ax.scatter(*(p.transpose(0, 1).cpu().detach().numpy()), s=1)
    ax.set_title(f"Points distribution iteration {i}")
    fig.savefig(f"results/iterations/iter_{i}")
    plt.close(fig)


# Plot the last position of the nodes
y = f(nn_approximator, x)
fig, ax = plt.subplots()
ax.set_title(f"Points distribution final iteration {i}")
ax.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy(), s=1)

# Plot the solution in a "dense" mesh
n_x = torch.linspace(X_INI, X_FIN, steps=1000 + 2, requires_grad=True, device=DEVICE)[
    1:-1
].reshape(-1)
n_x, _ = get_mesh(n_x, X_INI, X_FIN)

y = f(nn_approximator, n_x)
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), y.cpu().detach().numpy())
ax.scatter(n_x.cpu().detach().numpy(), y.cpu().detach().numpy(), s = 1)
ax.set_title("PINN solution")

exact_y = exact_solution(n_x, EPSILON)
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), exact_y.cpu().detach().numpy())
ax.set_title("Exact solution")

# PINN and exact solutions on one plot
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), exact_y.cpu().detach().numpy(), label="Exact")
ax.plot(n_x.cpu().detach().numpy(), y.cpu().detach().numpy(), label="PINN")
ax.legend()

error = y-exact_y
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), error.cpu().detach().numpy())
ax.scatter(n_x.cpu().detach().numpy(), error.cpu().detach().numpy())
ax.set_title("Error: NN_u - exact_solution")

# Draw the convergence plot
fig, ax = plt.subplots()
ax.semilogy(convergence_data.cpu().detach().numpy())
ax.set_title("Convergence")

plt.show()
