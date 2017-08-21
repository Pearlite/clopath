#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:56:56 2017

@author: koichirokajikawa
"""
import numpy as np
import matplotlib.pyplot as plt

# Activation function
def activation(x,method):
    if method=='exp':           # Exponential
        y=np.exp(x)
    elif method=='sigmoid':     # Sigmoid
        y=1/(1+np.exp(-(x)))
        # Can put several parameters as follows.
        # y=1/(1+np.exp(-(x-5)/8))
    elif method=='ReLU':        # ReLU
        y=np.maximum(0,x)
    return y

# Cluster index calculation based on weight matrix
def clustering_index(W_mat):
    # calculate a scalar as a clustering measure for a given weight matrix. This assumes an equal number of neurons in each cluster.
    mask = np.zeros(W_mat.shape) #create a mask of zeros and ones to index neurons in vs. neurons out of groups
    for i in range(C):
        mask[i*N_c:i*N_c+N_c, i*N_c:i*N_c+N_c] = 1;
    clustermean = np.mean(mask * W_mat) #get mean firing rate of all neurons within groups
    allmean  = np.mean(W_mat) #get mean firing rate of all neurons total
    return np.divide(clustermean, allmean)

## Parameters
# 1. Network structure
N=100       # number of neurons
C=5         # number of clusters
N_c=N//C    # number of neurons in a cluster
# 2. time constants: tau_r ≤ tau_t ≤ tau_w
[tau_r,tau_t,tau_w] = [5, 20, 1000]
[w_max, w_min]=[1,0]
# 3. Coefficients
#[w_c,s_c,ip_c]=[0.25,100,10]
#[w_c,s_c,ip_c]=[0.02,1.2,2]
[w_c,s_c,ip_c,th_c]=[0.02,1,10,1]

## Setup arrays
T_sec  = 5             # total time to simulate (sec)
T      = T_sec*10**3   # total time to simulate (msec)
dt     = 0.125         # simulation time step (msec)
time   = np.arange(0, T+dt, dt)    # time array
r      = np.zeros((N,len(time)))
r[:,0] = np.random.rand(N)
W      = np.zeros((N,N,len(time)))
W[:,:,0]= np.random.rand(N,N)*w_c
theta  = np.zeros((N,len(time)))
#theta[:,0]=np.random.rand(N)
#theta[:,0]=np.ones(N,)*np.random.rand(1)
theta[:,0]=activation((np.dot(W[:,:,0],r[:,0])),'sigmoid')**2
#theta[:,0]=np.ones(N,)

## Inputs
ip_time=10      # Duration of input ~ tau_r*10 (?)tau_theta
# Define input
ip_group=np.random.randint(0,C,int(T/ip_time))
#I=np.zeros((N, int(T/dt)))
I=np.ones((N, int(T/dt)))*(-10)
for k in range(len(ip_group)):
    I[N_c*ip_group[k]:N_c*(ip_group[k]+1),k*int(ip_time/dt):(k+1)*int(ip_time/dt)]=1*ip_c

# Learning
for t in range(len(time)-1):
    # Set diagonal entries as 0
    np.fill_diagonal(W[:,:,t],0)

    # Postsynaptic activity
    u=np.dot(W[:,:,t],r[:,t])+I[:,t]
    v=activation(u*s_c,'sigmoid')

    # rate update
    r[:,t+1]=r[:,t]+(-r[:,t]+v)*dt/tau_r

    # weight update
    r_post=r[:,t].reshape(-1,1)
#    r_pre=u.reshape(1,-1)
    r_pre=r[:,t].reshape(1,-1)

    W[:,:,t+1]=W[:,:,t]+(r_post*(r_post-theta[:,t].reshape(-1,1))*r_pre)*dt/tau_w
#    W[:,:,t+1]=np.minimum(W[:,:,t+1],w_max*np.ones((N,N)))
    W[:,:,t+1]=np.maximum(W[:,:,t+1],w_min*np.ones((N,N)))

    # theta update
    theta[:,t+1]=theta[:,t]+((r[:,t]**2)/th_c-theta[:,t])*dt/tau_t

print('Clustering strength is {:.3f}.'.format(clustering_index(W[:,:,-2])))

# l=-2
# plt.imshow(W[:,:,l])
# plt.colorbar()
# plt.show()

# print(W[-2,:,:].max())

# plt.plot()
# plt.plot(W[0,15,::100])
# plt.plot(W[0,25,::100])
# plt.plot(theta[0,::100])
# plt.plot(r[0,::100])

#inspect weight matrix - video over time
im = plt.imshow(W[:,:,0])
cb = plt.colorbar()
cb.set_clim(vmin=0, vmax=W[-2,:,:].max())
for i in range(1, len(time)-1, 1000):
    im.set_data(W[:,:,i])
    cb.draw_all()
    plt.pause(0.0001)
plt.show()
