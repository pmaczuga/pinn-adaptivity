# -*- coding: utf-8 -*-
import time

from src.adaptation import *
from src.advection_dominated_loss_function import *
import matplotlib.pyplot as plt
import torch

from scipy.integrate import quad

from src.error import *

def exact_(x):
    return exact_solution(x, 0.01)

def exact_grad_np(x):
    dx = dfdx(exact_, np_to_torch(x))
    return dx.detach().reshape(-1)

fun  = lambda x: (exact_grad_np(x))**2

# x = np.linspace(-1, 1, 1000)
# plt.plot(x, exact_grad_np(x))
# plt.show()

res, err = quad(fun, -1, 1)
print(f"Integration result: {res}")
print(f"Integration error: {err}")

nn_approximator = torch.load("results/data/pinn.pt")

l2 = L2_error(nn_approximator, exact_solution, EPSILON)
print(f"L2 error: {l2}")

h1 = H1_error(nn_approximator, exact_solution, EPSILON)
print(f"H1 error: {h1}")

def pinn_grad_np(x):
    dx = dfdx(nn_approximator, np_to_torch(x))
    return dx.detach().reshape(-1)

def pinn_np(x):
    return nn_approximator(np_to_torch(x)).detach().reshape(-1)

def exact_np(x):
    return exact_solution(np_to_torch(x), EPSILON).detach().reshape(-1)

x = np.linspace(-1, 1, 1000)
plt.plot(x, exact_np(x), label="exact")
plt.plot(x, pinn_np(x), "--",  label="PINN")
plt.legend()
plt.show()

x = np.linspace(-1, 1, 1000)
plt.plot(x, exact_grad_np(x), label="exact dx")
plt.plot(x, pinn_grad_np(x), "--", label="PINN dx")
plt.legend()
plt.show()
