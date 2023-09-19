import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import torch
from matplotlib import rc

from params import *
from src.adaptation import get_new_adapted_points
from src.advection_dominated_loss_function import (
    compute_loss,
    exact_solution,
    f_inter_loss,
)
from src.mesh_1D import get_mesh
from src.pinn_core import PINN, f, train_model

plt.rcParams["figure.dpi"] = 150
rc("animation", html="html5")

nn_approximator = torch.load("results/data/pinn.pt")
convergence_data = torch.load("results/data/convergence_data.pt")
point_data = torch.load("results/data/point_data.pt")
n_iters = torch.load("results/data/n_iters.pt")

x = torch.linspace(
    X_INI, X_FIN, steps=NUM_POINTS + 2, requires_grad=True, device=DEVICE
)[1:-1]
x, _ = get_mesh(x, X_INI, X_FIN)

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
fig.savefig(f"results/last_iter")

# Plot the solution in a "dense" mesh
n_x = torch.linspace(X_INI, X_FIN, steps=1000 + 2, requires_grad=True, device=DEVICE)[
    1:-1
].reshape(-1)
n_x, _ = get_mesh(n_x, X_INI, X_FIN)

y = f(nn_approximator, n_x)
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), y.cpu().detach().numpy())
ax.scatter(n_x.cpu().detach().numpy(), y.cpu().detach().numpy(), s=1)
ax.set_title("PINN solution")
fig.savefig(f"results/pinn_solution")

exact_y = exact_solution(n_x, EPSILON)
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), exact_y.cpu().detach().numpy())
ax.set_title("Exact solution")
fig.savefig(f"results/exact_solution")

# PINN and exact solutions on one plot
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), exact_y.cpu().detach().numpy(), label="Exact")
ax.plot(n_x.cpu().detach().numpy(), y.cpu().detach().numpy(), "--", label="PINN")
ax.legend()
fig.savefig(f"results/solutions")

error = y - exact_y
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), error.cpu().detach().numpy())
ax.set_title("Error: NN_u - exact_solution")
fig.savefig(f"results/error")

# Draw the convergence plot
fig, ax = plt.subplots()
ax.semilogy(convergence_data.cpu().detach().numpy())
ax.set_title("Convergence")
fig.savefig(f"results/convergence")

plt.show()
