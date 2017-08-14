import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

def sigmoid(x):
    return np.divide(1, (1+np.exp(sigmoid_coeff*x)))

def update(r_vec, W, I_vec): #r_vec is a vector (firing rates at previous timestep), W is a matrix (incoming weights to all neurons), I_vec is a vector (inputs to neurons at previous timestep)
    r_rep = np.matlib.repmat(r_vec, num_neurons, 1)
    assert W.shape == r_rep.shape
    x = np.sum(W*r_rep, axis=0) + I_scale*I_vec
    r_new = (r_vec + dt * (-r_vec + sigmoid(x)))/tau_c
    return r_new, x

def update_weights(r, W, x, threshold): #BCM. Sourcing diff eqs from https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0044-6
    threshold = threshold + dt * ((r*r - threshold)/tau_th)
    W = W + dt * ((r * x * (r - threshold))/ tau_w)
    np.fill_diagonal(W, 0) #no autapses
    return W, threshold

#scaling parameters; need to balance ratio
#rate code time constant, BCM threshold time constant, weight time constant, input scaling, sigmoid scaling, initial BCM threshold value
[tau_c, tau_th, tau_w, I_scale, W_scale, sigmoid_coeff, threshold] = [1, 20, 100, 0.5, 0.2, 1, 10] #clustering? Weight matrix looks weird.
# [tau_c, tau_th, tau_w, I_scale, W_scale, sigmoid_coeff, threshold] = [1, 20, 100, 0.5, 0.2, 1, 5]
# [tau_c, tau_th, tau_w, I_scale, W_scale, sigmoid_coeff, threshold] = [1, 20, 100, 0.5, 0.2, 1, 2] #no clustering

dt = 1
num_neurons = 100
runtime = 1000
num_groups = 5

W = np.random.rand(num_neurons, num_neurons) * W_scale
np.fill_diagonal(W, 0) #no autapses

# generate correlated inputs
idx = 0
global I
I = np.zeros((num_neurons, runtime/dt))
for i in range(num_groups):
    idx = i*(num_neurons/num_groups)
    I[idx:idx+20,:] = np.random.rand(1,runtime/dt) #matrix of input vectors
    # I[idx:idx+20,:] = np.random.binomial(1, 0.1, runtime/dt) #matrix of input vectors

r = np.zeros((num_neurons, runtime/dt))
r[:,0] = 0; #initial values of r

for t in range((runtime/dt)-1):
    r[:, t+1], x = update(r[:, t], W, I[:,t])
    W, threshold = update_weights(r[:, t+1], W, x, threshold)

plt.plot(np.transpose(r))
plt.show()

#inspect weight matrix
plt.imshow(W)
plt.show()
