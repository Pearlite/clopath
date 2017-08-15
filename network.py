import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

def update(r_vec, W, I_vec): #r_vec is a vector (firing rates at previous timestep), W is a matrix (incoming weights to all neurons), I_vec is a vector (inputs to neurons at previous timestep)
    r_rep = np.matlib.repmat(r_vec, num_neurons, 1)
    assert W.shape == r_rep.shape
    x = np.sum(W*r_rep, axis=0) + I_scale*I_vec
    r_new = (r_vec + dt * (-r_vec + np.exp(x)))/tau_c
    return r_new, x

def update_weights(r_vec, W_mat, x, threshold_vec): #BCM. Sourcing diff eqs from https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0044-6
    threshold_vec = threshold_vec + dt * ((r_vec*r_vec - threshold_vec)/tau_th)
    # print(threshold_vec)
    W_mat = W_mat + dt * ((r_vec * x * (r_vec - threshold_vec))/ tau_w)
    W_mat = W_mat.reshape(1, -1)
    W_mat = np.fmin(W_mat, np.ones(W_mat.shape)*W_max)
    assert(np.max(W_mat) <= W_max)
    W_mat = W_mat.reshape(num_neurons, num_neurons)
    np.fill_diagonal(W_mat, 0) #no autapses
    return W_mat, threshold_vec

#scaling parameters; need to balance ratio
#rate code time constant, BCM threshold time constant, weight time constant, input scaling, initial weight scaling, sigmoid scaling, initial BCM threshold value
[tau_c, tau_th, tau_w, I_scale, init_threshold] = [5, 10, 100, 1, 5]

dt = 0.1
num_neurons = 100
runtime = 1000
num_groups = 5
threshold = np.ones((num_neurons, int(runtime/dt)))
threshold[:,0] = init_threshold

W_scale = 0.02 #weak weights to start with; multiplication factor should be small
W_max = 1 #maximum weight value

W = np.empty((num_neurons, num_neurons, int(runtime/dt)))
W[:,:,0] = np.random.rand(num_neurons, num_neurons) * W_scale
np.fill_diagonal(W[:,:,1], 0) #no autapses

min_input_duration = 100 #in timesteps

idx = 0
# global I
I = np.zeros((num_neurons, int(runtime/dt)))
which_group = np.random.randint(0,5, int((runtime/dt)/min_input_duration))
idx = [i * num_neurons/num_groups for i in range(num_groups)]
for i in range(len(which_group)):
    I[idx[which_group[i]]:idx[which_group[i]]+(num_neurons/num_groups), i*min_input_duration:(i+1)*min_input_duration] = 1

r = np.zeros((num_neurons, int(runtime/dt)))
r[:,0] = 0; #initial values of r

for t in range((int(runtime/dt))-1):
    r[:, t+1], x = update(r[:, t], W[:,:,t], I[:,t])
    W[:,:,t+1], threshold[:,t+1] = update_weights(r[:, t+1], W[:,:,t], x, threshold[:,t])

plt.plot(np.transpose(r))
plt.show()

#inspect weight matrix
plt.subplot(3,1,1)
plt.imshow(W[:,:,0])
plt.subplot(3,1,2)
plt.imshow(W[:,:,int(runtime/dt)/3])
plt.subplot(3,1,3)
plt.imshow(W[:,:,int(runtime/dt)-1])
plt.colorbar()
plt.show()
