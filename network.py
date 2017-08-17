import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

def update(r_vec, W_mat, I_vec): # update the firing rate. r_vec is a vector (firing rates at previous timestep), W is a matrix (incoming weights to all neurons), I_vec is a vector (inputs to neurons at previous timestep)
    r_rep = np.matlib.repmat(r_vec, num_neurons, 1)
    assert W_mat.shape == r_rep.shape
    x = np.sum(W_mat*r_rep, axis=0) + I_scale*I_vec # elementwise multiplication of presynaptic firing rate with the associated weights, and adding external input. Together, this is the input vector x to the postsynaptic neuron.
    r_post = (r_vec + dt * (-r_vec + np.exp(x/1.2)))/tau_c #calculate postsynaptic firing rate
    return r_post, x #return new firing rate, as well as input vector x

def update_weights(r_vec, W_mat, x, threshold_vec): #BCM. Sourcing diff eqs from https://mathematical-neuroscience.springeropen.com/articles/10.1186/s13408-017-0044-6
    # update threshold theta (different for each neuron)
    threshold_vec = threshold_vec + dt * ((r_vec*(r_vec - threshold_vec))/tau_th)

    # update weight matrix; reshape a few things to make sure I get the outer product rather than the dot product
    x = x.reshape(1,-1)
    r_vec = r_vec.reshape(-1,1)
    threshold_vec = threshold_vec.reshape(-1,1)
    W_mat = W_mat + dt * ((r_vec * (r_vec - threshold_vec) * x)/ tau_w)
    threshold_vec = np.squeeze(threshold_vec)

    # make sure weights are within bounds; reshape some things because np.fmin, np.fmax only work on arrays, not matrices
    W_mat = W_mat.reshape(1, -1)
    W_mat = np.fmax(W_mat, np.zeros(W_mat.shape)*W_min)
    W_mat = np.fmin(W_mat, np.ones(W_mat.shape)*W_max)
    assert(np.max(W_mat) <= W_max)
    assert(np.min(W_mat) >= W_min)
    W_mat = W_mat.reshape(num_neurons, num_neurons)

    # remove autapses - doesn't seem to make a difference in practice
    # np.fill_diagonal(W_mat, 0)

    return W_mat, threshold_vec

def calc_clustering_index(W_mat):
    # calculate a scalar as a clustering measure. This assumes an equal number of neurons in each cluster.
    groupsize = num_neurons/num_groups
    mask = np.zeros(W_mat.shape) #create a mask of zeros and ones to index neurons in vs. neurons out of groups
    for i in range(num_groups):
        mask[i*groupsize:i*groupsize+groupsize, i*groupsize:i*groupsize+groupsize] = 1;
    clustermean = np.mean(mask * W_mat) #get mean of all neurons within groups
    nonclustermean = np.mean((np.ones(mask.shape) - mask) * W_mat) #get mean of all neurons outside of groups
    return np.divide(clustermean, nonclustermean) #return clustering index scalar

# scaling parameters; need to balance ratio
# rate code time constant, BCM threshold time constant, weight time constant, input scaling, initial weight scaling,  initial BCM threshold value
[tau_c, tau_th, tau_w, I_scale, init_threshold] = [5, 50, 1000, 2, 0] # time constants in ms

#set simulation parameters
dt = 0.1
num_neurons = 100
runtime = 1000 # in ms
num_groups = 5
threshold = np.zeros((num_neurons, int(runtime/dt)))
threshold[:,0] = init_threshold

W_scale = 0.02 #weak weights to start with; multiplication factor should be small
[W_max, W_min] = [1, 0] #weight bounds; we don't want inhibitory neurons

W = np.empty((num_neurons, num_neurons, int(runtime/dt)))
W[:,:,0] = np.random.rand(num_neurons, num_neurons) * W_scale
np.fill_diagonal(W[:,:,1], 0) #no autapses

min_input_duration = int(20 / dt) #in ms

idx = 0
I = np.zeros((num_neurons, int(runtime/dt)))
which_group = np.random.randint(0,5, int((runtime/dt)/min_input_duration))
idx = [i * num_neurons/num_groups for i in range(num_groups)]
for i in range(len(which_group)):
    # I[idx[which_group[i]]:idx[which_group[i]]+(num_neurons/num_groups), i*min_input_duration:(i+1)*min_input_duration] = 1 * np.random.rand(num_neurons/num_groups, min_input_duration)
    I[idx[which_group[i]]:idx[which_group[i]]+(num_neurons/num_groups), i*min_input_duration:(i+1)*min_input_duration] = 1

r = np.zeros((num_neurons, int(runtime/dt)))
r[:,0] = 0; #initial values of r
# r[:,0] = np.random.rand(num_neurons)

clustering = np.zeros(int(runtime/dt)-1)

for t in range((int(runtime/dt))-1):
    clustering[t] = calc_clustering_index(W[:,:,t])
    r[:, t+1], x = update(r[:, t], W[:,:,t], I[:,t])
    W[:,:,t+1], threshold[:,t+1] = update_weights(r[:, t+1], W[:,:,t], x, threshold[:,t])

plt.plot(np.transpose(r))
plt.show()

plt.plot(clustering)
plt.show()

# #inspect weight matrix - still images
# plt.subplot(3,1,1)
# plt.imshow(W[:,:,0])
# plt.colorbar()
# plt.subplot(3,1,2)
# plt.imshow(W[:,:,int(runtime/dt)/3])
# plt.colorbar()
# plt.subplot(3,1,3)
# plt.imshow(W[:,:,int(runtime/dt)-1])
# plt.colorbar()
# plt.show()

# plt.plot(np.transpose(threshold)) #show threshold over time for all neurons
# plt.show()

#inspect weight matrix - video
im = plt.imshow(W[:,:,0])
cb = plt.colorbar()
cb.set_clim(vmin=0, vmax=0.05)
for i in range(1, int(runtime/dt), 1000):
    im.set_data(W[:,:,i])
    cb.draw_all()
    plt.pause(0.0001)
plt.show()
