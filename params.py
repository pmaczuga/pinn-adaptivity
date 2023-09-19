import torch

from params import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

X_INI = -1.0
LENGTH = 2.0
X_FIN = X_INI + LENGTH

NUM_POINTS = 100

EPSILON = 0.05  # <-HERE W CHANGE THE epsilon (=0.1 is working, =0.01 is not working)

MAX_ITERS = 1000
NUMBER_EPOCHS = 200
LAYERS = 3
NEURONS = 15
LEARNING_RATE = 0.005

# Tolerance
TOL = 1e-5