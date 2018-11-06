#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy import stats
from pandas import *

#### Model w/o relationships ###################################################
flag = 1 # 0 = NoR model; 1 = Rmodel

### Parameters #################################################################
# scales: tons, years, MXNia, hours, trips
tmax = 27 # model run, years
b0 = -16.49 # SST trend
b1 = 0.025 # SST trend
b2 = 9 # SST trend
b3 = 0.091 # SST trend
l1 = -0.0059 # q, slope
l2 = 0.1882 # q, intersect
a1 = 1/(np.exp(30.823998124274-(b0+b1*(30+2015)))) # migration trigger
f = 0 # intercept of trader cooperation
d = 5 # slope of trader cooperation
K = 1208770 # carrying capacity
g = 1.4 # population growth rate
gamma = 49200 # maximum demand
beta = 0.0736 # slope of demand-price function
c_p = 1776.25 # cost of processing
c_t = 156076110 # cost of fishing

B_h = 7.203 # hours per fisher
B_f = 2 # fisher per panga
h1 = 2E-10 # scale E
h2 = 0.6596 # scale E

### Variables ##################################################################
tau = np.zeros(tmax) # temperature
q = np.zeros(tmax) # catchability squid population
ML = np.zeros(tmax) # mantle length
y_S = np.zeros(tmax) # distance of squid migration from initial fishing grounds
R_tt = np.zeros(tmax) # trader cooperation
S = np.zeros(tmax) # size of the squid population
Escal = np.zeros(tmax) # scale effort
E = np.zeros(tmax) # fishing effort
C = np.zeros(tmax) # squid catch
p_e = np.zeros(tmax) # export price
p_escal = np.zeros(tmax) # export price
p_min = np.zeros(tmax) # minimum wage
p_f = np.zeros(tmax) # price for fishers
RF = np.zeros(tmax) # revenue of fishers
RT = np.zeros(tmax) # revenue of traders
RA = np.zeros(tmax) # revenue all fishery

### Initial values #############################################################
tau[0] = 42. # isotherm depth
q[0] = 0.01 # squid catchability
y_S[0] = 0.5 # proportion of migrated squid
R_tt[0] = 0.5 # trader cooperation
S[0] = 1208770 # size of the squid population
E[0] = 1. # fishing effort
C[0] = 120877 # squid catch
p_e[0] = 99366 # max p_e comtrade
p_f[0] = 15438 # max p_f datamares


################################################################################
###########################  MODEL FILE  #######################################
################################################################################

#### Load dataset  #############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3_data.xlsx', sheetname='Sheet1')
#! load columns
y = df1['year'] #
pe = df1['pe_MXNiat'] #
pf = df1['pf_MXNiat'] #
ct = df1['C_t'] #
ssh = df1['essh_avg'] #
ml = df1['ML'] #
ys = df1['y_S'] #

### New max time
tmax = len(y)

### Define Model ###############################################################
def model(b0, b1, b2, b3, l1, l2, a1, f, d, g, K, h1, h2, gamma, beta, c_p, c_t, flag):
    for t in np.arange(1,tmax):
    # isotherm depth
        # sst trend
        tau[t]= b0 +b1 *(t+2015) +b2 *np.cos(t+2015) + b3 *np.sin(t+2015)

        # catchability sst
        q[t]= l1 *tau[t] +l2
        if q[t] > 0.1:
            q[t] = 0.1
            print "q high"
        elif q[t] < 0:
            q[t] = 0
            print "q low"

        # migration of squid
        y_S[t] = a1 *np.exp(tau[t]-(b0 +b1*(t+2015))) # sst trend
        if y_S[t] > 1:
            y_S[t] = 1
            print "yS high"
        elif y_S[t] < 0.01:
            y_S[t] = 0.01
            print "yS low"

        # trader cooperation
        R_tt[t]= f+ np.exp(-d* y_S[t])

        # squid population
        S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        # fishing effort
        Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -c_t *E[t-1]
        # fishing effort scaled
        E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]
        if E[t] > 1:
            E[t] = 1
            print "E high"
        elif E[t] < 0:
            E[t] = 0
            print "E low"

        # catch
        C[t] = q[t] *E[t] *S[t]

        if C[t] <= 0:
            C[t] = 1
            print "C low"

        # export price
        p_e[t] = gamma* (C[t])**(-beta)

        if p_e[t]>= 99366:
            p_e[t]= 99366
            print "pe high"

        #### switch between models ####
        if flag == 0:
            # price for fishers
            p_f[t] = p_e[t] -c_p
        if flag == 1:
            # minimum wage
            p_min[t]= (c_t *E[t])/C[t] #  MXN/ton
            # price for fishers
            p_f[t] = (p_e[t] -c_p) *(1-R_tt[t]) +R_tt[t] *p_min[t]

        if p_f[t] >= (p_e[t] -c_p):
            p_f[t] = p_e[t] -c_p
            print "pf high"

        # revenue of fishers
        RF[t] = C[t] *p_f[t] -c_t *E[t]
        # revenue of traders
        RT[t] = C[t] *p_e[t] -RF[t] -c_p
        # revenue of all fishery
        RA[t] = C[t] *p_e[t]

        #print t, tau[t], ML[t], q[t], y_S[t], S[t], E[t], C[t], p_e[t], p_f[t]
    return tau, ML, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA


################################################################################
#############################  RUN MODEL FILE  #################################
################################################################################

#### Model w/o relationships ###################################################
flag = 1 # must be 1!!! 1 = Rmodel

##### Run the model ############################################################
b1 = np.arange(0.013,0.027,0.0001)
b2 = np.arange(5.25,8.75,0.025)

##### Initiate arrays ##########################################################
cat = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save catches in each time period of each simulation
pri = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save prices for fishers in each time period of each simulation
mar = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save market prices in each time period of each simulation
gap1 = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save price gap mean in each time period of each simulation
gap2 = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save price gap stdv in each time period of each simulation
tem = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save tau in each time period of each simulation
mig = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save migrate squid in each time period of each simulation
cco = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save catchability in each time period of each simulation
pop = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save squid population in each time period of each simulation
eff = np.zeros((b1.shape[0],b2.shape[0])) # matrix to save effort in each time period of each simulation

for i in np.arange(0,b1.shape[0]):
    for j in np.arange(0,b2.shape[0]):
        tau, ML, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA = model(b0, b1[i], b2[j], b3, l1, l2, a1, f, d, g, K, h1, h2, gamma, beta, c_p, c_t, flag)
        gap1[i,j]= np.mean(p_f/p_e)
        gap2[i,j]= np.std(p_f/p_e)

        cat[i,j]= np.mean(C)
        pri[i,j]= np.mean(p_f)
        mar[i,j]= np.mean(p_e)

        tem[i,j]= np.mean(tau)
        cco[i,j]= np.mean(q)
        pop[i,j]= np.mean(S)
        eff[i,j]= np.mean(E)


################################################################################
################################  PLOT FILE  ###################################
################################################################################

##### Plot 1 ###################################################################
## define dimensions
y = b1 #  y axis
x = b2 #  x axis
z = gap1 #  output data
## sub plot
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.1,left=0.1,right=0.9)
ax = fig1.add_subplot(gs[0,0])
pcObject = ax.pcolormesh(x,y,z)
plt.pcolormesh(x, y, z, cmap="Spectral")
# both axis
plt.tick_params(axis=1, which='major', labelsize=12)
## set y-axis
ax.set_ylabel('Trend $b1$', fontsize = 22)
plt.ylim(0.013,0.026)
## set x-axis
ax.set_xlabel('Amplitude $b2$', fontsize = 22)
plt.xlim(5.25,8.5)
## colorbar
cb = plt.colorbar()
cb.set_label((r'mean price gap $\frac{P_f}{P_m}$'), rotation=270, labelpad=40, fontsize = 22)
# plt.clim([0,1])
## save and show
# fig1.savefig("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R_b1b2_mean.png",dpi=500)
plt.show()

##### Plot 2 ###################################################################
## define dimensions
y = b1 #  y axis
x = b2 #  x axis
z = gap2 #  output data
## sub plot
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.1,left=0.1,right=0.9)
ax = fig1.add_subplot(gs[0,0])
pcObject = ax.pcolormesh(x,y,z)
plt.pcolormesh(x, y, z, cmap="Spectral")
# both axis
plt.tick_params(axis=1, which='major', labelsize=12)
## set y-axis
ax.set_ylabel('Trend $b1$', fontsize = 22)
plt.ylim(0.013,0.026)
## set x-axis
ax.set_xlabel('Amplitude $b2$', fontsize = 22)
plt.xlim(5.25,8.5)
## colorbar
cb = plt.colorbar()
cb.set_label((r'std price gap $\frac{P_f}{P_m}$'), rotation=270, labelpad=40, fontsize = 22)
# plt.clim([0,1])
## save and show
# fig1.savefig("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R_b1b2_std.png",dpi=500)
plt.show()

################################################################################
#############################  RUN TIMESERIES ##################################
################################################################################
#
# OUT1 = np.zeros(tau.shape[0])
# OUT2 = np.zeros(tau.shape[0])
# OUT3 = np.zeros(tau.shape[0])
# OUT4 = np.zeros(tau.shape[0])
# OUT5 = np.zeros(tau.shape[0])
# OUT6 = np.zeros(tau.shape[0])
# OUT7 = np.zeros(tau.shape[0])
# OUT8 = np.zeros(tau.shape[0])
# OUT9 = np.zeros(tau.shape[0])
#
# for i in np.arange(0,1):
#         tau, ML, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA = model(b0, b1, b2, b3, l1, l2, a1, f, d, g, K, h1, h2, gamma, beta, c_p, c_t, flag)
#         OUT1[i]= p_f[i]
#         OUT2[i]= C[i]
#         OUT3[i]= tau[i]
#         OUT4[i]= y_S[i]
#         OUT5[i]= q[i]
#         OUT6[i]= S[i]
#         OUT7[i]= E[i]
#         OUT8[i]= p_e[i]
#
#
# ################################################################################
# ###########################  PLOT FILE  ########################################
# ################################################################################
#
# ### font ######################################################################
# hfont = {'fontname':'Helvetica'}
#
# #####! PLOT MODEL  #############################################################
# # Three subplots sharing both x/y axes
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# # axis 1
# ax1.plot(tau, label = "tau", color="orange")
# ax1.set_title('Temperature')
# ax1.legend(loc="best")
# # axis 2
# ax2.plot(y_S, label = "migration", color="orange")
# ax2.plot(q, label = "catchability", color="steelblue")
# ax2.plot(E, label = "effort", color="red")
# # ax2.plot(G, label = "pay gap", color="green")
# ax2.legend(loc="best")
# # axis 3
# ax3.plot(C, label = "catch", color="orange")
# ax3.plot(S, label = "population", color="steelblue")
# ax3.legend(loc="best")
# # axis 4
# ax4.plot(p_f, label = "price for fishers", color="orange")
# ax4.plot(p_e, label = "market price", color="steelblue")
# ax4.legend(loc="best")
# # Fine-tune figure; make subplots close to each other and hide x ticks for
# # all but bottom plot.
# f.subplots_adjust(hspace=0.2)
# plt.show()
