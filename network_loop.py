#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:56:56 2017

@author: koichirokajikawa, marritzuure
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

# Cluster index calculation based on weight matrix and neurons per cluster
def clustering_index(W_mat, C, N_c):
    # calculate a scalar as a clustering measure for a given weight matrix. This assumes an equal number of neurons in each cluster.
    mask = np.empty(W_mat.shape)
    mask[:] = np.NAN
    for i in range(C):
        mask[i*N_c:i*N_c+N_c, i*N_c:i*N_c+N_c] = 1; #create a mask of NaNs and ones to index neurons in vs. neurons out of groups
    clustermean = np.nanmean(np.multiply(mask, W_mat)) #get mean firing rate of all neurons within groups
    allmean  = np.mean(W_mat) #get mean firing rate of all neurons total
    return np.divide(clustermean, allmean)

def weight_matrix_video(W, time, wmtitle=''):
    #inspect weight matrix - video over time
    im = plt.imshow(W[:,:,0])
    cb = plt.colorbar()
    plt.title(wmtitle)
    cb.set_clim(vmin=0, vmax=W[-2,:,:].max())
    for i in range(1, len(time)-1, 1000):
        im.set_data(W[:,:,i])
        cb.draw_all()
        plt.pause(0.0001)
    plt.show()

class Parameters:
    def __init__(self):
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

        # create object containing all initial parameters needed for simulation
        [self.N, self.C, self.N_c, self.T, self.time, self.dt, self.W, self.r, self.I, self.theta, self.tau_r, self.tau_w, self.tau_t, self.w_min, self.w_max, self.s_c, self.th_c, self.ip_c, self.ip_time] = N, C, N_c, T, time, dt, W, r, I, theta, tau_r, tau_w, tau_t, w_min, w_max, s_c, th_c, ip_c, ip_time

def run_simulation(p):
    # Simulation
    for t in range(len(p.time)-1):
        # Set diagonal entries as 0
        np.fill_diagonal(p.W[:,:,t],0)

        # Postsynaptic activity
        u=np.dot(p.W[:,:,t],p.r[:,t])+p.I[:,t]
        v=activation(u*p.s_c,'sigmoid')

        # rate update
        p.r[:,t+1]=p.r[:,t]+(-p.r[:,t]+v)*p.dt/p.tau_r

        # weight update
        r_post=p.r[:,t].reshape(-1,1)
        r_pre=p.r[:,t].reshape(1,-1)
        p.W[:,:,t+1]=p.W[:,:,t]+(r_post*(r_post-p.theta[:,t].reshape(-1,1))*r_pre)*p.dt/p.tau_w
        p.W[:,:,t+1]=np.minimum(p.W[:,:,t+1],p.w_max*np.ones((p.N,p.N)))
        p.W[:,:,t+1]=np.maximum(p.W[:,:,t+1],p.w_min*np.ones((p.N,p.N)))

        # theta update
        p.theta[:,t+1]=p.theta[:,t]+((p.r[:,t]**2)/p.th_c-p.theta[:,t])*p.dt/p.tau_t

    clustering = clustering_index(p.W[:,:,-2], p.C, p.N_c)
    return p, clustering

def run_simulation_fix_weights(p):
    # Simulation
    for t in range(len(p.time)-1):
        # Postsynaptic activity
        u=np.dot(p.W,p.r[:,t])+p.I[:,t]
        v=activation(u*p.s_c,'sigmoid')
        # rate update
        p.r[:,t+1]=p.r[:,t]+(-p.r[:,t]+v)*p.dt/p.tau_r
    return p

# # plot sigmoid for illustrative purposes
# y = np.empty(len(range(-10,10)))
# counter = 0
# for x in range(-10, 10):
#     y[counter] = activation(x, 'sigmoid')
#     counter = counter+1
# plt.plot(y)
# plt.xticks(range(len(y)), range(-10, 10))
# plt.show()

### RUN SIMULATIONS ###
# # to simulate once:
# p = Parameters()
# [p, clustering] = run_simulation(p)
# plt.imshow(p.I[:,0:399])
# plt.xlabel('Time')
# plt.ylabel('Neurons')
# # plt.colorbar()
# plt.show()
# plt.imshow(p.W[:,:,-2])
# plt.colorbar()
# plt.show()
# print('Clustering strength is {:.3f}.'.format(clustering))
# weight_matrix_video(p.W, p.time)

#### SIMULATION: TESTING WHETHER WEIGHTS DO ANYTHING
# p = Parameters()
# # p.W = np.ones((p.N, p.N))
# p.I = np.ones((p.I.shape)) * -53
# p.I[20,:] = 1000 #set really strong input to ten neuron
# # p = run_simulation_fix_weights(p)
# p.W = np.ones((p.W.shape)) * 0.2
# # p = run_simulation(p)
# [p, clustering] = run_simulation(p)
# plt.plot(np.transpose(p.r))
# plt.show()
# plt.plot(p.r[:,-2])
# plt.show()
# plt.imshow(p.W[:,:,-2])
# plt.show()


#### SIMULATION: MORE TESTING WHETHER WEIGHTS DO ANYTHING
# wire clusters together with weight 1
# p = Parameters()
# test_W = np.zeros((p.N, p.N))
# for i in range(p.C):
#     test_W[i*p.N_c:i*p.N_c+p.N_c, i*p.N_c:i*p.N_c+p.N_c] = 1; #create a mask of NaNs and ones to index neurons in vs. neurons out of groups
# plt.imshow(test_W)
# plt.show()
# p.W = test_W
#
# #set high input to one neuron. Value of negative input is chosen to not automatically drive neurons up.
# p.I = np.ones(p.I.shape) * -11
# p.I[11,:] = 100
#
# # run simulation
# p = run_simulation_fix_weights(p)
#
# #plot firing rates. If weights work, should see more than 1 neuron going to 1.
# plt.plot(np.transpose(p.r))
# plt.show()
#
# #plot firing rates at end of simulation. If weights work, should see 20 highly active neurons. If weights don't work, just 1.
# plt.plot(p.r[:,-2])
# plt.show()
#
# #Getting the sense that the weights genuinely don't work...

#### SIMULATION: ENDLESS TESTING WHETHER WEIGHTS DO ANYTHING
# # wire 50 neurons to the other 50 with weight 1
# p = Parameters()
# test_W = np.ones((p.N, p.N))
# test_W = np.tril((test_W)) #lower triangle is now 1s, rest 0s
# np.fill_diagonal(test_W, 0)
# plt.imshow(test_W)
# plt.show()
# p.W = test_W
#
# # #set high input to ~20 neuron. Value of negative input is chosen to not automatically drive neurons up.
# p.I = np.ones(p.I.shape) * -11
# p.I[1:20,:] = 100
#
# # run simulation
# p = run_simulation_fix_weights(p)
#
# #plot firing rates. If weights work, should see more than 1 neuron going to 1.
# plt.plot(np.transpose(p.r))
# plt.show()
#
# #plot firing rates at end of simulation. If weights work, should see 20 highly active neurons. If weights don't work, just 1.
# plt.plot(p.r[:,-2])
# plt.show()
#
# #oh. weights do work I guess. Using triu vs. tril determines whether entire network is active or just the neurons receiving input.

### RUN SIMULATIONS ###
# to simulate once: (changing I to incorporate Ornstein-Uhlenbeck process-generated noise)
# p = Parameters()
# tau_OU = 20 #Ornstein-Uhlenbeck time constant
# c_OU = 10 # noise strength
# mu = -10 # long-term mean
# p.I=np.zeros((p.N, int(p.T/p.dt)))
# np.random.seed = 1 #remove effects of stochasticity between simulations
# for t in range(int(p.T/p.dt)-1):
#     r1 = np.kron(np.random.randn(p.C),np.ones(p.N_c))
#     p.I[:,t+1] = p.I[:,t] - mu
#     p.I[:,t+1] = p.I[:,t+1] * np.exp(-p.dt/tau_OU) + np.sqrt((c_OU * tau_OU*0.5)*(1-(np.exp(-p.dt/tau_OU))**2)) * r1
#     p.I[:,t+1] = p.I[:,t+1] + mu
#
# # # calculate correlations between different inputs
# # corrs = np.empty((p.C, p.C))
# # for c1 in range(p.C):
# #     for c2 in range(p.C):
# #         corrs[c1, c2] = np.corrcoef(p.I[c1*p.N_c], p.I[c2*p.N_c])[1,0]
# # plt.imshow(corrs)
# # plt.colorbar()
# # plt.show()
#
# # plot sample of input to network
# plt.imshow(p.I[:,0:199])
# plt.colorbar()
# plt.title('OU process-generated input (μ = {})'.format(mu))
# plt.xlabel('time')
# plt.ylabel('neurons')
# plt.show()
#
# # run simulation
# [p, clustering] = run_simulation(p)
#
# # plot mean firing rates and mean theta
# plt.plot(p.r.mean(0), label='Mean firing rate')
# plt.xlabel('Time')
# plt.plot(p.theta.mean(0), label='Mean θ')
# plt.legend()
# plt.show()
#
# wmtitle = 'Weights after OU input with μ = {} (CI = {:.3f})'.format(mu, clustering)
# print(wmtitle)
# weight_matrix_video(p.W, p.time, wmtitle)
#
# #plot W within cluster and outside cluster over time
# plt.plot(p.W[25,30,:], label='Within cluster')
# plt.plot(p.W[25,0,:], label='Outside cluster')
# plt.title('Weights given OU input with μ = {} (CI = {:.3f})'.format(mu, clustering))
# plt.legend()
# plt.show()

#### SIMULATION: CORRELATED INPUTS, SUMMED OU PROCESSES ####
# # 1. Generate lots of OU processes: one for every neuron plus one for every cluster
# p = Parameters()
#
# tau_OU = 20 #Ornstein-Uhlenbeck time constant
# c_OU = 10 # noise strength
# mu = -10 # long-term mean
#
# OUs = np.empty((p.N + p.C, int(p.T/p.dt)))
# OUs[:,0] = (np.random.randn(p.N + p.C)-1)*c_OU
# for t in range(int(p.T/p.dt)-1):
#     r1 = np.random.randn(p.N + p.C)
#     OUs[:,t+1] = OUs[:,t] - mu
#     OUs[:,t+1] = OUs[:,t+1] * np.exp(-p.dt/tau_OU) + np.sqrt((c_OU * tau_OU*0.5)*(1-(np.exp(-p.dt/tau_OU))**2)) * r1
#     OUs[:,t+1] = OUs[:,t+1] + mu
#
# # plt.imshow(OUs[:,1:400]) # visual inspection, sanity check
# # plt.show()
#
# # 2. Generate weighted summations of OU processes (for example, 0.8 cluster + 0.2 individual noise)
# w1 = 0.3 #cluster input weighting
# w2 = 1 - w1 #individual noise weighting
# OUn = np.empty((p.N, int(p.T/p.dt))) #OUs for clusters
# for n in range(p.N):
#     cl = int(np.floor(n/p.N_c)) #cluster that current neuron belongs to
#     OUn[n] = w1 * OUs[cl] + w2 * OUs[n+p.C]
#
# plt.imshow(OUn[:,1:400]) # visual inspection, sanity check
# plt.title('Noisy input (cluster input Cw={}, individual noise Nw={:.2})'.format(w1, w2))
# plt.xlabel('Time')
# plt.ylabel('Neuron')
# plt.show()
#
# # 3. Assign weighted summations to input
# p.I = OUn
#
# # 4. Run simulation
# [p, clustering] = run_simulation(p)
# # print('Clustering strength is {:.3f}.'.format(clustering))
# wmtitle = 'Weights after OU input with Cw={}, Nw={:.2} (CI = {:.3f})'.format(w1, w2, clustering)
# weight_matrix_video(p.W, p.time, wmtitle)

### SIMULATION: CORRELATED INPUTS, MULTIPLE WEIGHTING VALUES ###
# # 1. Generate lots of OU processes: one for every neuron plus one for every cluster
# p = Parameters()
#
# tau_OU = 20 #Ornstein-Uhlenbeck time constant
# c_OU = 10 # noise strength
# mu = -10 # long-term mean
#
# OUs = np.empty((p.N + p.C, int(p.T/p.dt)))
# OUs[:,0] = (np.random.randn(p.N + p.C)-1)*c_OU
# for t in range(int(p.T/p.dt)-1):
#     r1 = np.random.randn(p.N + p.C)
#     OUs[:,t+1] = OUs[:,t] - mu
#     OUs[:,t+1] = OUs[:,t+1] * np.exp(-p.dt/tau_OU) + np.sqrt((c_OU * tau_OU*0.5)*(1-(np.exp(-p.dt/tau_OU))**2)) * r1
#     OUs[:,t+1] = OUs[:,t+1] + mu
#
# # 2. Generate weighted summations of OU processes (for example, 0.8 cluster + 0.2 individual noise)
# w1_range = np.arange(1, 0, -0.1) # cluster input weighting
# clustering = np.empty(len(w1_range))
# weightmatrices = np.empty((p.N, p.N, len(w1_range)))
# ci = 0 #counter
# for w1 in w1_range:
#     print(ci) #progress indicator for long simulations
#     w2 = 1 - w1 # individual noise weighting
#     OUn = np.empty((p.N, int(p.T/p.dt))) #OUs for clusters
#     for n in range(p.N):
#         cl = int(np.floor(n/p.N_c)) #cluster that current neuron belongs to
#         OUn[n] = w1 * OUs[cl] + w2 * OUs[n+p.C]
#     # 3. Assign weighted summations to input
#     p.I = OUn
#     # 4. Run simulation
#     [p, clustering[ci]] = run_simulation(p)
#     weightmatrices[:,:,ci] = p.W[:,:,-2]
#     ci = ci + 1
#
# plt.plot(clustering)
# plt.ylabel('Clustering index')
# plt.xlabel('Cw')
# plt.xticks(range(len(w1_range)), w1_range)
# plt.show()
#
# for q in range(len(w1_range)):
#     plt.imshow(weightmatrices[:,:,q])
#     plt.colorbar()
#     plt.show()

### SIMULATION: FIXED WEIGHTS, CORRELATED INPUTS, PLOT FIRING RATES PER CLUSTER
# # 1. Generate OU processes
# p = Parameters()
#
# tau_OU = 20 #Ornstein-Uhlenbeck time constant
# c_OU = 10 # noise strength
# mu = -10 # long-term mean
#
# OUs = np.empty((p.N + p.C, int(p.T/p.dt)))
# OUs[:,0] = (np.random.randn(p.N + p.C)-1)*c_OU
# for t in range(int(p.T/p.dt)-1):
#     r1 = np.random.randn(p.N + p.C)
#     OUs[:,t+1] = OUs[:,t] - mu
#     OUs[:,t+1] = OUs[:,t+1] * np.exp(-p.dt/tau_OU) + np.sqrt((c_OU * tau_OU*0.5)*(1-(np.exp(-p.dt/tau_OU))**2)) * r1
#     OUs[:,t+1] = OUs[:,t+1] + mu
#
# # 2. Recombine OU processes for learning
# w1 = 0.8 #cluster input weighting
# w2 = 1 - w1 #individual noise weighting
# OUn = np.empty((p.N, int(p.T/p.dt))) #OU per neuron
# for n in range(p.N):
#     cl = int(np.floor(n/p.N_c)) #cluster that current neuron belongs to
#     OUn[n] = w1 * OUs[cl] + w2 * OUs[n+p.C]
#
# # 3. Set input
# p.I = OUn
#
# # 4. Run simulation to get weight matrix out
# [p, clustering] = run_simulation(p)
# learned_W = p.W[:,:,-2]
#
# # 5. Add a small input increase ( = signal) to one cluster
# s = 10
# OUn = np.empty((p.N, int(p.T/p.dt)))
# target = np.random.randint(0,p.C) #select one cluster to receive signal
# for n in range(p.N):
#     cl = int(np.floor(n/p.N_c)) #cluster that current neuron belongs to
#     OUn[n] = OUs[n+p.C]
#     if cl == target:
#         OUn[n] = OUn[n] + s
#
# plt.imshow(OUn[:,1:400]) # visual inspection, sanity check
# plt.show()
#
# # 6. Set input
# p = Parameters()
# p.I = OUn
# p.W = learned_W
#
# # 7. Run simulation with fixed weights
# p = run_simulation_fix_weights(p)
#
# # plt.imshow(p.r[:,1000:2000])
# # plt.show()
#
# #plot firing rates per cluster
# mean_rates = np.empty((len(range(p.C)), 1))
# for cl in range(p.C):
#     # print('Plotting cluster {}, neuron {} until {}.'.format(cl, cl*p.N_c, (cl+1)*p.N_c))
#     plt.plot(np.mean(p.r[cl*p.N_c:(cl+1)*p.N_c], 0), label='Cluster {}'.format(cl+1))
#     mean_rates[cl] = np.mean(np.mean(p.r[cl*p.N_c:(cl+1)*p.N_c], 0))
# print(mean_rates)
# print(np.mean(mean_rates))
# [ratio] = np.divide(mean_rates[target], np.mean(mean_rates))
# plt.legend()
# plt.title('Target cluster = {}, $μ_t / μ_a$ = {:.3}'.format(target+1, ratio))
# plt.xlabel('Time')
# plt.ylim((0, 1))
# plt.ylabel('Mean firing rates per cluster')
# plt.show()
#
# print('mu_target/mu_all = {}'.format(ratio))
#
# plt.imshow(p.W)
# plt.show()

#### SIMULATIONS: TIME OFFSETS IN THE OU PROCESSES
# #1. Generate OU processes that are longer than needed for the simulation
# p = Parameters()
#
# tau_OU = 20 #Ornstein-Uhlenbeck time constant
# c_OU = 10 # noise strength
# mu = -10 # long-term mean
#
# length = int((p.T/p.dt)*1.5)
# OUs = np.empty((p.N + p.C, length))
# OUs[:,0] = (np.random.randn(p.N + p.C)-1)*c_OU
# for t in range(length-1):
#     r1 = np.random.randn(p.N + p.C)
#     OUs[:,t+1] = OUs[:,t] - mu
#     OUs[:,t+1] = OUs[:,t+1] * np.exp(-p.dt/tau_OU) + np.sqrt((c_OU * tau_OU*0.5)*(1-(np.exp(-p.dt/tau_OU))**2)) * r1
#     OUs[:,t+1] = OUs[:,t+1] + mu
#
# #2. Generate the input vectors by recombining and time-shifting within-cluster processes
# w1 = 0.8 #cluster input weighting
# w2 = 1 - w1 #individual noise weighting
# offset = 8
# OUn = np.empty((p.N, int(p.T/p.dt))) #OU per neuron
# for n in range(p.N):
#     cl = int(np.floor(n/p.N_c)) #cluster that current neuron belongs to
#     in_cl = np.mod(n, p.N_c) #this is the xth neuron in the cluster
#     OUn[n] = w1 * OUs[cl, in_cl*offset:in_cl*offset+int(p.T/p.dt)] + w2 * OUs[n+p.C, 0:int(p.T/p.dt)]
# # for n in range(p.N):
# #     OUn[n] = OUs[0, n*offset:n*offset+int(p.T/p.dt)]
#
# #3. Set input
# p.I = OUn
# plt.imshow(p.I[:,0:399])
# plt.show()
#
# #4. Run simulation
# [p, clustering] = run_simulation(p)
# weight_matrix_video(p.W, p.time)

#### SIMULATION: PATTERN COMPLETION - Generate degraded input and see how the network responds
# # 1. Generate OU processes
# p = Parameters()
#
# tau_OU = 20 #Ornstein-Uhlenbeck time constant
# c_OU = 10 # noise strength
# mu = -10 # long-term mean
#
# OUs = np.empty((p.N + p.C, int(p.T/p.dt)))
# OUs[:,0] = (np.random.randn(p.N + p.C)-1)*c_OU
# for t in range(int(p.T/p.dt)-1):
#     r1 = np.random.randn(p.N + p.C)
#     OUs[:,t+1] = OUs[:,t] - mu
#     OUs[:,t+1] = OUs[:,t+1] * np.exp(-p.dt/tau_OU) + np.sqrt((c_OU * tau_OU*0.5)*(1-(np.exp(-p.dt/tau_OU))**2)) * r1
#     OUs[:,t+1] = OUs[:,t+1] + mu
#
# # 2. Recombine OU processes for learning
# w1 = 0.8 #cluster input weighting
# w2 = 1 - w1 #individual noise weighting
# OUn = np.empty((p.N, int(p.T/p.dt))) #OU per neuron
# for n in range(p.N):
#     cl = int(np.floor(n/p.N_c)) #cluster that current neuron belongs to
#     OUn[n] = w1 * OUs[cl] + w2 * OUs[n+p.C]
#
# # 3. Set input
# p.I = OUn
#
# # 4. Run simulation to get weight matrix out
# [p, clustering] = run_simulation(p)
# learned_W = p.W[:,:,-2]
#
# # 5. Generate degraded patterns ( = set some inputs to a constant -10, based on degradation factor)
# degrade = 0.25 # proportion of inputs in target cluster to drop
#  #shuffle OUs indices to ensure I don't have the exact same inputs as in the learning phase
# shuffle = np.random.permutation(len(OUs))
# OUs = OUs[shuffle]
#
# OUn = np.ones((p.N, int(p.T/p.dt))) - 10 #OU per neuron
# target = np.random.randint(0,p.C) #select one cluster to receive (degraded) signal; others will receive noise
# for n in range(p.N):
#     cl = int(np.floor(n/p.N_c)) #cluster that current neuron belongs to
#     OUn[n] = OUs[n+p.C]
#     if cl == target:
#         rnum = np.random.rand()
#         if rnum < degrade:
#             OUn[n] = np.ones((1, int(p.T/p.dt)))*-10
#
# # 4. Set inputs and weight matrix
# p = Parameters()
# p.W = learned_W
# p.I = OUn
#
# plt.imshow(p.I[:,0:399])
# plt.show()
#
# # 5. Run fixed-weight simulation
# p = run_simulation_fix_weights(p)
#
# #plot firing rates per cluster
# mean_rates = np.empty((len(range(p.C)), 1))
# for cl in range(p.C):
#     # print('Plotting cluster {}, neuron {} until {}.'.format(cl, cl*p.N_c, (cl+1)*p.N_c))
#     plt.plot(np.mean(p.r[cl*p.N_c:(cl+1)*p.N_c], 0), label='Cluster {}'.format(cl+1))
#     mean_rates[cl] = np.mean(np.mean(p.r[cl*p.N_c:(cl+1)*p.N_c], 0))
# print(mean_rates)
# print(np.mean(mean_rates))
# [ratio] = np.divide(mean_rates[target], np.mean(mean_rates))
# plt.legend()
# plt.title('Target cluster = {}, $μ_t / μ_a$ = {:.3}'.format(target+1, ratio))
# plt.xlabel('Time')
# plt.ylim((0, 1))
# plt.ylabel('Mean firing rates per cluster')
# plt.show()

# # ### RUN SIMULATIONS ###
# # to simulate often with different parameters (in this example, with noisy inputs):
# noise_amplitude_range = range(0,10,1)
# clustering = np.empty(len(noise_amplitude_range)) #initialize array to hold clustering outputs
# ci = 0 #initialize counter for clustering index matrix
# for noise_amplitude in noise_amplitude_range:
#     p = Parameters()     # initialize
#     p.I = p.I + noise_amplitude*np.random.standard_normal(p.I.shape)*p.ip_c #add both positive and negative noise
#     [p, clustering[ci]] = run_simulation(p)     #run simulation
#     ci = ci+1 #update counter
# # plot outcomes
# plt.plot(clustering)
# plt.xlabel('Noise amplitude')
# plt.xticks(range(len(noise_amplitude_range)), noise_amplitude_range)
# plt.ylabel('Clustering index')
# plt.show()

### RUN SIMULATIONS ###
# # to simulate often with different parameters (in this example, with different values of tau_th given OU process):
# tau_t_range = range(25,250,25)
# clustering = np.empty(len(tau_t_range)) #initialize array to hold clustering outputs
# p = Parameters() #extracting simulation duration to initialize mean_thetas array
# mean_thetas = np.empty((len(tau_t_range), len(p.time)))
# mean_rates = np.empty((len(tau_t_range), len(p.time)))
# ci = 0 #initialize counter for clustering index matrix
# for tau_t in tau_t_range:
#     print(ci) #progress indicator for long simulations
#     p = Parameters()
#     p.tau_t = tau_t
#     tau_OU = 20 #Ornstein-Uhlenbeck time constant
#     c_OU = 10 # noise strength
#     mu = 0 # long-term mean
#     p.I=np.zeros((p.N, int(p.T/p.dt)))
#     np.random.seed = 1 #remove effects of stochasticity between simulations
#     for t in range(int(p.T/p.dt)-1):
#         r1 = np.kron(np.random.randn(p.C),np.ones(p.N_c))
#         p.I[:,t+1] = p.I[:,t] - mu
#         p.I[:,t+1] = p.I[:,t+1] * np.exp(-p.dt/tau_OU) + np.sqrt((c_OU * tau_OU*0.5)*(1-(np.exp(-p.dt/tau_OU))**2)) * r1
#         p.I[:,t+1] = p.I[:,t+1] + mu
# # plt.imshow(p.I[:,0:199])
# # plt.colorbar()
# # plt.title('OU process-generated input (μ = {})'.format(mu))
# # plt.xlabel('time')
# # plt.ylabel('neurons')
# # plt.show()
#     [p, clustering[ci]] = run_simulation(p)
#     mean_thetas[ci,:] = p.theta.mean(0)
#     mean_rates[ci,:] = p.r.mean(0)
#     ci = ci+1
#
# plt.plot(np.transpose(mean_thetas), label=tau_t_range)
# # plt.plot(np.transpose(mean_rates))
# plt.xlabel('time')
# plt.ylabel('mean plasticity threshold θ')
# plt.show()
# # plt.legend()
#
# plt.plot(clustering)
# plt.show()

# # ### RUN SIMULATIONS ###
# # to simulate with two different sets of parameters (in this case, OU mu and tau_th):
# tau_t_range = range(10,100,20)
# mu_range = range(1,-10,-4)
# clustering = np.empty((len(tau_t_range), len(mu_range)))# initialize matrix to hold clustering outputs
# [ci, cj] = 0, 0 #initialize counters for clustering index matrix
# for tau_t in tau_t_range:
#     for mu in mu_range:
#         print('{},{}'.format(ci, cj)) # progress indicator for long simulations
#
#         # initialize
#         p = Parameters()
#
#         # change parameters to loop through
#         p.tau_t = tau_t
#
#         tau_OU = 20 #Ornstein-Uhlenbeck time constant
#         c_OU = 10 # noise strength
#         p.I=np.zeros((p.N, int(p.T/p.dt)))
#         np.random.seed = 1 #remove effects of stochasticity between simulations
#         for t in range(int(p.T/p.dt)-1):
#             r1 = np.kron(np.random.randn(p.C),np.ones(p.N_c))
#             p.I[:,t+1] = p.I[:,t] - mu
#             p.I[:,t+1] = p.I[:,t+1] * np.exp(-p.dt/tau_OU) + np.sqrt((c_OU * tau_OU*0.5)*(1-(np.exp(-p.dt/tau_OU))**2)) * r1
#             p.I[:,t+1] = p.I[:,t+1] + mu
#
#         [p, clustering[cj, ci]] = run_simulation(p)
#         ci = ci+1
#     ci = 0
#     cj = cj+1
#
# plt.imshow(clustering)
# plt.title("Clustering index for different combinations of τθ, μ")
# plt.xlabel("μ")
# plt.xticks(range(len(mu_range)), mu_range)
# plt.ylabel("τθ")
# plt.yticks(range(len(tau_t_range)), tau_t_range)
# plt.colorbar()
# plt.show()

# # ### RUN SIMULATIONS ###
# # to simulate with two different sets of parameters (in this example, tau_th and tau_r):
# tau_t_range = range(10,100,10)
# tau_r_range = range(1,10,1)
# clustering = np.empty((len(tau_t_range), len(tau_r_range)))# initialize matrix to hold clustering outputs
# [ci, cj] = 0, 0 #initialize counters for clustering index matrix
# for tau_t in tau_t_range:
#     for tau_r in tau_r_range:
#         # initialize
#         p = Parameters()
#
#         # change parameters to loop through
#         p.tau_t = tau_t
#         p.tau_r = tau_r
#         [p, clustering[cj, ci]] = run_simulation(p)
#         ci = ci+1
#     ci = 0
#     cj = cj+1
#
# plt.imshow(clustering)
# plt.title("Clustering index for different combinations of tau_t, tau_r")
# plt.xlabel("tau_r")
# plt.xticks(range(len(tau_r_range)), tau_r_range)
# plt.ylabel("tau_t")
# plt.yticks(range(len(tau_t_range)), tau_t_range)
# plt.colorbar()
# plt.show()

## old plotting code down here
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
