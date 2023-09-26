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
from src.error import H1_error, L2_error
from src.mesh_1D import get_mesh
from src.pinn_core import PINN, f, train_model

plt.rcParams["figure.dpi"] = 150
rc("animation", html="html5")

nn_approximator = torch.load("results/data/pinn.pt")
convergence_data = torch.load("results/data/convergence_data.pt")
point_data = torch.load("results/data/point_data.pt")
n_iters = torch.load("results/data/n_iters.pt")
exec_time = torch.load("results/data/exec_time.pt")

l2 = L2_error(nn_approximator, exact_solution, EPSILON)
h1 = H1_error(nn_approximator, exact_solution, EPSILON)

# Write params and results to text file
with open("results/result.txt", "w") as file:
    file.write(f"DEVICE = {DEVICE}\n")
    file.write(f"NUM_BASE_POINTS = {NUM_BASE_POINTS}\n")
    file.write(f"NUM_MAX_POINTS = {NUM_MAX_POINTS}\n")
    file.write(f"EPSILON = {EPSILON}\n")
    file.write(f"NUMBER_EPOCHS = {NUMBER_EPOCHS}\n")
    file.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
    file.write(f"LAYERS = {LAYERS}\n")
    file.write(f"NEURONS = {NEURONS}\n")
    file.write(f"ADAPTATION = {ADAPTATION}\n")
    file.write(f"TOL = {TOL}")
    file.write(f"\n")
    file.write(f"Time = {exec_time}\n")
    file.write(f"L2 error = {l2}\n")
    file.write(f"H1 error = {h1}\n")

# for i, p in enumerate(point_data):
#     fig, ax = plt.subplots()
#     ax.scatter(*(p.transpose(0, 1).cpu().detach().numpy()), s=1)
#     ax.set_title(f"Points distribution iteration {i}")
#     fig.savefig(f"results/iterations/iter_{i}")
#     plt.close(fig)


# Plot the solution in a "dense" mesh
n_x = torch.linspace(X_INI, X_FIN, steps=1000 + 2, requires_grad=True, device=DEVICE)[
    1:-1
].reshape(-1)
n_x, _ = get_mesh(n_x, X_INI, X_FIN)

y = f(nn_approximator, n_x)
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), y.cpu().detach().numpy())
ax.scatter(n_x.cpu().detach().numpy(), y.cpu().detach().numpy(), s=1)
ax.set_title(f"PINN solution, eps={EPSILON}")
fig.savefig(f"results/pinn_solution")


# Plot exact solution
exact_y = exact_solution(n_x, EPSILON)
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), exact_y.cpu().detach().numpy())
ax.set_title(f"Exact solution, eps={EPSILON}")
fig.savefig(f"results/exact_solution")

# PINN and exact solutions on one plot
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), exact_y.cpu().detach().numpy(), label="Exact")
ax.plot(n_x.cpu().detach().numpy(), y.cpu().detach().numpy(), "--", label="PINN")
ax.legend()
ax.set_title("PINN and exact, eps={}, L2 = {:.2e}, H1={:.2e}".format(EPSILON, l2, h1))
fig.savefig(f"results/solutions")

error = y - exact_y
fig, ax = plt.subplots()
ax.plot(n_x.cpu().detach().numpy(), error.cpu().detach().numpy())
ax.set_title("Error: NN_u - exact_solution, eps={}, L2 = {:.2e}, H1={:.2e}".format(EPSILON, l2, h1))
fig.savefig(f"results/error")

# Draw the convergence plot
fig, ax = plt.subplots()
ax.semilogy(convergence_data.cpu().detach().numpy())
ax.set_title(f"Convergence, time = {exec_time} s")
fig.savefig(f"results/convergence")

plt.show()
