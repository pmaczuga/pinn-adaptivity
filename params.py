import torch

from params import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

X_INI = -1.0
X_FIN = 1.0

NUM_BASE_POINTS = 20
NUM_MAX_POINTS = 200

EPSILON = 0.1  # <-HERE WE CHANGE THE epsilon (=0.1 is working, =0.01 is not working)

MAX_ITERS = 1000
NUMBER_EPOCHS = 1000
LAYERS = 3
NEURONS = 15
LEARNING_RATE = 0.005

ADAPTATION = False

# Tolerance
TOL = 1e-4