# this file generates the price difference (export-beach price) for two interventions: increase demand and increase competition between traders 
# Dr Laura Gabriele Elsler

#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy.stats as st
from pandas import *

### Parameters #################################################################
# scales: tons, years, MXNia, hours, trips
tmax = 30 # model run, years
b1 = 41.750 # isotherm depth
b2 = -5.696 # isotherm depth
b3 = 16.397 # isotherm depth
n1 = -22.239 # ML, slope
n2 = 49.811 # ML, intersect
l1 = -0.0028 # q, slope
l2 = 0.1667 # q, intersect
a1 = 1/3.4E7 # proportion of migrating squid, where 3.4E7 max(e^(tau-b1))
K = 1208770 # carrying capacity
g = 1.4 # population growth rate
gamma = 49200 # maximum demand
beta = 0.0736 # slope of demand-price function
w_m = 13355164 # min wage per hour all fleet
c_p = 1776.25 # cost of processing
c_t = 156076110 # cost of fishing
m = 156076110 # cost per unit of transport all boats, MXN/trip
f = 1 # l of fuel per trip

B_h = 7.203 # hours per fisher
B_f = 2 # fisher per panga
h1 = 2E-10 # scale E
h2 = 0.6596 # scale E
flag = 3 # this gives an error if its not been changed previously to the right model

### Variables ##################################################################
tau = np.zeros(tmax) # temperature
q = np.zeros(tmax) # catchability squid population
ML = np.zeros(tmax) # mantle length
y_S = np.zeros(tmax) # distance of squid migration from initial fishing grounds
R_tt = np.zeros(tmax) # trader cooperation
S = np.zeros(tmax) # size of the squid population
c_t = np.zeros(tmax) # cost of transport
Escal = np.zeros(tmax) # scale effort
E = np.zeros(tmax) # fishing effort
C = np.zeros(tmax) # squid catch
p_e = np.zeros(tmax) # export price
p_escal = np.zeros(tmax) # export price
p_min = np.zeros(tmax) # minimum wage
p_f = np.zeros(tmax) # price for fishers
R = np.zeros(tmax) # revenue of fishers

### Initial values #############################################################
tau[0] = 42. # isotherm depth
q[0] = 0.01 # squid catchability
y_S[0] = 0.5 # proportion of migrated squid
R_tt[0] = 0.5 # trader cooperation
S[0] = 1208770 # size of the squid population
c_t[0] = m *f # fleet cost of transport
E[0] = 1. # fishing effort
C[0] = 120877 # squid catch
p_e[0] = 99366 # max p_e comtrade
p_f[0] = 15438 # max p_f datamares

################################################################################
###############################  MODEL FILE  ###################################
################################################################################

### Define parameter ###########################################################
Rtt = 0.

### Define Model ###############################################################
def model(b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag, Rtt):
    for t in np.arange(1,tmax):
        # isotherm depth
        tau[t]= b1 +b2 *np.cos(t) + b3 *np.sin(t)
        # mantle length
        ML[t]= n1 *tau[t] + n2
        # catchability
        q[t]= l1 *tau[t] +l2
        # migration of squid
        y_S[t] = a1 *np.exp(tau[t]-b1)
        # squid population
        S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]
        # cost of transport
        c_t[t]= m *f # I decided to use fixed costs over migration, that equally well/better predicted catches over m* (y_S[t]); (source: LabNotesSquid, April 11)
        # fishing effort
        Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -c_t[t-1] *(E[t-1]/(B_h*B_f)) # c_t is per trip so we need to upscale E hr > fisher > trip
        # fishing effort scaled
        E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]
        # catch
        C[t] = q[t] *E[t] *S[t]
        # export price
        p_e[t] = gamma* (C[t])**(-beta)

        #### switch between models ####
        if flag == 0:
            # price for fishers
            p_f[t] = p_e[t] -c_p
        if flag == 1:
            # minimum wage
            p_min[t]= (E[t] *w_m)/C[t]
            # price for fishers
            p_f[t] = (p_e[t] -c_p) *(1-Rtt) +Rtt *p_min[t]

        # revenue of fishers
        R[t] = C[t] *p_f[t] - c_t[t-1] *(E[t-1]/(B_h+B_f))
    return tau, ML, q, y_S, S, c_t, E, C, p_e, p_f, R


################################################################################
#############################  RUN MODEL FILE  #################################
################################################################################

#### Model w/o relationships ###################################################
flag = 1 # must be 1!!! 1 = Rmodel

##### Run the model ############################################################
gamma = np.arange(10000.,110000.,1000.)
Rtt = np.arange(0.,1.,0.01)

OUT = np.zeros((gamma.shape[0], Rtt.shape[0]))
OUT1 = np.zeros((gamma.shape[0], Rtt.shape[0]))

for i in np.arange(0,gamma.shape[0]):
    for j in np.arange(0,gamma.shape[0]):
        tau, ML, q, y_S, S, c_t, E, C, p_e, p_f, R = model(b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma[i], beta, c_p, w_m, flag, Rtt[j])
        OUT[i,j]= np.mean(p_f)
        OUT1[i,j]= np.mean(p_e)


################################################################################
################################  PLOT FILE  ###################################
################################################################################

##### Plot stuff ###############################################################
## define dimensions
y = gamma #  y axis
x = Rtt #  x axis
z = OUT/OUT1 #  output data
## sub plot
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.1,left=0.1,right=0.9)
ax = fig1.add_subplot(gs[0,0])
pcObject = ax.pcolormesh(x,y,z)
plt.pcolormesh(x, y, z, cmap="Spectral")
# both axis
plt.tick_params(axis=1, which='major', labelsize=12)
## set y-axis
ax.set_ylabel('demand $\gamma $', fontsize = 22)
plt.ylim(1E4,1.09E5)
## set x-axis
ax.set_xlabel('trader cooperation $R$', fontsize = 22)
plt.xlim(0,0.9)
## colorbar
cb = plt.colorbar()
cb.set_label(r'$\frac{P_f}{P_e}$', rotation=0, labelpad=15, fontsize = 28, fontweight='bold')
# plt.clim([0,1])
## save and show
# fig1.savefig("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R4_20180411.png",dpi=500)
plt.show()
