
import torch

from src.gaussian_quadrature_1D import d_transform
    

def get_random_points(xini, xfin, order):
    gp = torch.rand(order, requires_grad=True, device=xini.device, dtype=torch.float32)
    gp = 2*gp-1.0
    return d_transform(gp, torch.ones_like(gp), xini, xfin)[1]

def get_equispaced_points(xini, xfin, order):
    gp = torch.linspace(-1, 1, order+2, requires_grad=True, device=xini.device, dtype=torch.float32)[1:-1]
    return d_transform(gp, torch.ones_like(gp), xini, xfin)[1]

def get_latin_hypercube(xini, xfin, order):
    device = xini.device
    xini = xini.item()
    xfin = xfin.item()
    ranges = torch.linspace(xini, xfin, order+1, device=device, dtype=torch.float32)
    a = ranges[:-1]
    b = ranges[1:]
    points = torch.rand(order, requires_grad=True, device=device, dtype=torch.float32)
    return points * (b-a) + a

def get_id_points(xini, xfin, order):
    return torch.tensor([xini, xfin])