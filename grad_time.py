# -*- coding: utf-8 -*-
import time

from src.adaptation import *
import matplotlib.pyplot as plt
import torch

x = torch.linspace(0, 1, 10)
loss = lambda x: torch.exp(-(x*6.0 - 3.0)**2)

start_time = time.time()
new_x = refine(x, loss, 100, 0.5)
end_time = time.time()
print(f" Refinement took {end_time - start_time} seconds")

plt.scatter(new_x.detach(), loss(new_x.detach()), s=1)
plt.show()