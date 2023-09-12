# -*- coding: utf-8 -*-
import time

from src.pinn_core import *

x = torch.linspace(0, 1, 10, requires_grad=True)

start = time.time()

for _ in range(20):
    y = torch.exp(x * 2 - 1) * x + torch.sin(x) * torch.cos(x * x)
    der = df(y, x)

end = time.time()

print(end - start)
