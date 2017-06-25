__author__ = 'ryanarjun'
import numpy as np


# Activation Function
def sigmoidFunc(x, derivative=False):
    if derivative:
        return x * (1-x)
    return 1 / (1 + np.exp(-x))


# Input data for the X
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Expected output
y = np.array([[0],
             [0],
             [1],
             [1]])

# Give the generated numbers the same seed
np.random.seed(1)

# Synapses
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# Training step of the Neural
for j in xrange(50000):

    # Each layer will forward propagate the data
    layer0 = X
    layer1 = sigmoidFunc(np.dot(layer0, syn0))
    layer2 = sigmoidFunc(np.dot(layer1, syn1))

    # Compare the results
    layer2_error = y - layer2
    if j % 10000 == 0:
        print "Error" + str(np.mean(np.abs(layer2_error)))

    # Get the delta
    layer2_delta = layer2_error * sigmoidFunc(layer2, derivative=True)
    layer1_error = layer2_delta.dot(syn1.T)
    layer1_delta = layer1_error * sigmoidFunc(layer1, derivative=True)

    # Update weights of the synapses
    syn1 += layer1.T.dot(layer2_delta)
    syn0 += layer0.T.dot(layer1_delta)

# Print the results of the matrix
print "Output after training"
print layer2