import math
import torch
import numpy as np
from scipy.integrate import quad
from src.advection_dominated_loss_function import exact_solution

from params import *
from src.pinn_core import dfdx

def np_to_torch(v: torch.tensor):
    return torch.tensor(v, dtype=torch.float32, requires_grad=True).cpu().reshape(-1, 1)

def L2_error(pinn, exact, epsilon):
    def pinn_np(x):
        return pinn(np_to_torch(x)).detach().reshape(-1)

    def exact_np(x):
        return exact(np_to_torch(x), epsilon).detach().reshape(-1)
    
    up   = lambda x: (pinn_np(x) - exact_np(x))**2
    dwon = lambda x: exact_np(x)**2

    up_int   = quad(up, -1, 1)[0]
    down_int = quad(dwon, -1, 1)[0]

    return math.sqrt(up_int / down_int)

def H1_error(pinn, exact, epsilon):
    def pinn_np(x):
        return pinn(np_to_torch(x)).detach().reshape(-1)

    def pinn_grad_np(x):
        dx = dfdx(pinn, np_to_torch(x))
        return dx.detach().reshape(-1)

    def exact_np(x):
        return exact(np_to_torch(x), epsilon).detach().reshape(-1)
    
    def exact_(x):
        return exact(x, epsilon)
    
    def exact_grad_np(x):
        dx = dfdx(exact_, np_to_torch(x))
        return dx.detach().reshape(-1)

    up1  = lambda x: (pinn_np(x) - exact_np(x))**2
    up2  = lambda x: (pinn_grad_np(x) - exact_grad_np(x))**2
    down = lambda x: exact_np(x)**2

    up1_int  = quad(up1, 0, 1)[0]
    up2_int  = quad(up2, 0, 1)[0]
    down_int = quad(down, 0, 1)[0]

    return math.sqrt((up1_int + up2_int) / down_int)
