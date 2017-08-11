import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.divide(1, (1+np.exp(-x/10)))

def update(r, W, I): #r is a vector (firing rates at previous timestep), W is a vector (incoming weights to all neurons), I is a vector (inputs to neurons at given time)
    return r + dt * (-r + sigmoid(np.sum(W*np.matlib.repmat(r, num_neurons, 1), axis=0) + I)) #need to take r'th column from W to multiply with all previous values of r, for each r separately. How?

# def BCM():

dt = 1
num_neurons = 100
runtime = 1000
num_groups = 5

W = np.divide(np.random.rand(num_neurons, num_neurons), 10) #random weight matrix. 0-1. TODO: save for reproducibility. TODO: sparse matrix?
np.fill_diagonal(W, 0) #no autapses

# generate correlated inputs
idx = 0
I = np.zeros((num_neurons, runtime/dt))
for i in range(num_groups):
    idx = i*(num_neurons/num_groups)
    # I[idx:idx+20,:] = np.random.rand(1,runtime/dt) #matrix of input vectors
    I[idx:idx+20,:] = np.random.binomial(1, 0.1, runtime/dt) #matrix of input vectors
    print(I[idx,1:10])

r = np.zeros((num_neurons, runtime/dt))
r[:,0] = 0; #initial values of r

for t in range((runtime/dt)-1):
    r[:, t+1] = update(r[:, t], W, I[:,t])

plt.plot(np.transpose(r))
plt.show()

# # if I did this neuron by neuron:
# for t in range(runtime/dt):
#     for n in range(num_neurons):
#         r[n, t+1] = update(r[n,t], W, I[n,t])
