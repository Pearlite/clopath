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

def weight_matrix_video(W, time, wmtitle):
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
# print('Clustering strength is {:.3f}.'.format(clustering))
# weight_matrix_video(p.W, p.time)

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
