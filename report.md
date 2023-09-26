
```
NUM_BASE_POINTS = 20
NUM_MAX_POINTS = 200

EPSILON = 0.01 

MAX_ITERS = 1000
NUMBER_EPOCHS = 3000
LAYERS = 3
NEURONS = 15
LEARNING_RATE = 0.005
```
Does not converge. When loss goes down to 0.00001 after changing points it gets to couple of thousands. Happened couple of times.

```
NUM_BASE_POINTS = 20
NUM_MAX_POINTS = 200

EPSILON = 0.01

MAX_ITERS = 1000
NUMBER_EPOCHS = 10000
LAYERS = 3
NEURONS = 30
LEARNING_RATE = 0.005
```
Training was stuck in local minima with `loss = 0.66` (near this value) for 10 iterations (100 000 epochs total). Did not converge.

# Working adaptation

```
NUM_BASE_POINTS = 20
NUM_MAX_POINTS = 200

EPSILON = 0.01  # <-HERE W CHANGE THE epsilon (=0.1 is working, =0.01 is not working)

MAX_ITERS = 1000
NUMBER_EPOCHS = 1000
LAYERS = 3
NEURONS = 15
LEARNING_RATE = 0.005

# Tolerance
TOL = 1e-4
```
With Adamax optimizer. Same optimizer is kept all the time (I do not recreate it each iteration). 129 iterations.

# Same as above but NO adaptation

```
NUM_BASE_POINTS = 20
NUM_MAX_POINTS = 200

EPSILON = 0.01  # <-HERE W CHANGE THE epsilon (=0.1 is working, =0.01 is not working)

MAX_ITERS = 1000
NUMBER_EPOCHS = 1000
LAYERS = 3
NEURONS = 15
LEARNING_RATE = 0.005

# Tolerance
# I changed that to make it as good as possible
# Training ends when loss < TOL (just that)
TOL = 1e-6 
```

It works worse!!! Yey
100 iterations

# Notes
Adam don't know how to get out of the local minima. Creating new instance in every iteration HELPS A LOT. However it does not work well for adaptation. Adamax on the other hand does get out of local minima. And keeping same optimizer makes adaptation actually work.
