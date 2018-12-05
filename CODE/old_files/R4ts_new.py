#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy.stats as st
from pandas import *

#### model w/o relationships ###################################################
flag = 1 # DEFAULT 1 for BEM+
timeInt = 30 # time of intervention
intervention = 2 # 0 = no intervention, 1 = demand int, 2 = competition int REMEMBER TO ACTIVATE THE INTERVENTIONS BELOW
demand = 0 # demand intervention, 0 = inactive, 1 = active
competition = 1 # competition intervention, 0 = inactive, 1 = active

# noint = 000, demand = 110, competition = 201

tmax = 60 # model run, years

### Variables ##################################################################
tau = np.zeros(tmax) # temperature
ML = np.zeros(tmax) # mantle length
q = np.zeros(tmax) # catchability squid population
y_S = np.zeros(tmax) # distance of squid migration from initial fishing grounds
R_tt = np.zeros(tmax) # trader cooperation
S = np.zeros(tmax) # size of the squid population
Escal = np.zeros(tmax) # scale effort
E = np.zeros(tmax) # fishing effort
C = np.zeros(tmax) # squid catch
p_e = np.zeros(tmax) # export price
p_min = np.zeros(tmax) # minimum wage
p_f = np.zeros(tmax) # price for fishers
RFn = np.zeros(tmax) # revenue of fishers normalized
RF = np.zeros(tmax) # revenue of fishers
RT = np.zeros(tmax) # revenue of traders
RA = np.zeros(tmax) # revenue all fishery
G = np.zeros(tmax) # pay gap between fishers and traders

### Parameters #################################################################
# scales: tons, years, MXNia, hours, trips
b0 = -16.49 # SST trend
b1 = 0.02 # SST trend
b2 = 6.779 # SST trend
b3 = 0.091 # SST trend
l1 = -0.0059 # q, slope
l2 = 0.1882 # q, intersect
qc = 0.1 # catchability constant
a1 = 1/(np.exp(30.823998124274-(b0+b1*(30+2015)))) # migration trigger
tau[0] = 30. # for SST
c_t = 107291548 # cost of fishing
f = 0 # intercept of trader cooperation
d = 1 # slope of trader cooperation
K = 1208770 # carrying capacity
g = 1.4 # population growth rate
gamma = 49200 # maximum demand
beta = 0.0736 # slope of demand-price function
c_p = 1776.25 # cost of processing

h1 = 2E-10 # scale E
h2 = 0.6596 # scale E

### Initial values #############################################################
q[0] = 0.05 # squid catchability
y_S[0] = 0.5 # proportion of migrated squid
R_tt[0] = 0.5 # trader cooperation
S[0] = 610075 # size of the squid population, mean
E[0] = 0.5 # fishing effort
C[0] = 60438 # squid catch mean
p_e[0] = 52035 # mean p_e comtrade, rounded
p_f[0] = 8997 # mean p_f datamares, rounded

################################################################################
###########################  MODEL FILE  #######################################
################################################################################

#### Parameters (constants)
d1 = 0.05
d2 = 0.05

### Define Model ###############################################################
def model(competition, demand, d1, d2, b0, b1, b2, b3, l1, l2, qc, a1, d, f, g, K, h1, h2, gamma, beta, c_p, c_t, flag):
    for t in np.arange(1,tmax):
        tau[t]= b0 +b1 *(t+2015) +b2 *np.cos(t+2015) + b3 *np.sin(t+2015)

        # mantle length and catchability
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
        elif y_S[t] < 0:
            y_S[t] = 0
            print "yS low"

        # trader cooperation
        if t <= timeInt:
            R_tt[t] = (1-y_S[t])
        else:
            if competition == 0:
                R_tt[t] = (1-y_S[t])
            if competition == 1:
                R_tt[t] = (1-y_S[t]) *(1/(1+(t-timeInt) *d2))

        #### switch between models ####
        if flag == 0: # squid population BEM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - qc *E[t-1] *S[t-1]
        if flag >= 1: # squid population MLM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        #### switch between models ####
        if flag == 0: # effort BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -c_t *E[t-1] # c_t is per trip so we need to upscale E hr > fisher > trip
            # fishing effort scaled
            E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]

        if flag >= 1: # effort MLM
            Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -c_t *E[t-1] # c_t is per trip so we need to upscale E hr > fisher > trip
            # fishing effort scaled
            E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]

        if E[t] > 1:
            E[t] = 1
            print "E high"
        elif E[t] < 0:
            E[t] = 0
            print "E low"

        #### switch between models ####
        if flag == 0: # catch BEM
            C[t] = qc *E[t] *S[t]
        if flag >= 1: # catch MLM
            C[t] = q[t] *E[t] *S[t]

        if C[t] <= 0: # avoid infinity calculations
            C[t]= 1

        # export price
        if t <= timeInt:
            p_e[t] = gamma* (C[t])**(-beta)
        else:
            if demand == 0:
                p_e[t] = gamma* (C[t])**(-beta)
            if demand == 1:
                p_e[t] = (gamma *(1+ d1 *(t -timeInt))) *(C[t])**(-beta)

        if p_e[t] >= 99366:
            p_e[t] = 99366
            print "pe high"

        #### switch between models ####
        if flag == 0: # BEM
            # price for fishers
            p_f[t] = p_e[t] -c_p
        if flag == 2: # BLM
            p_f[t] = p_e[t] -c_p    # price for fishers
        if flag == 1: # MLM
            p_min[t]= (c_t *E[t])/C[t] # ->must be MXN/ton to fit into p_f calc
            # price for fishers
            p_f[t] = (p_e[t] -c_p) *(1-R_tt[t]) +R_tt[t] *p_min[t]

        if p_f[t] >= (p_e[t] -c_p): # limit price of fishers
            p_f[t] = (p_e[t] -c_p)

        # revenue of fishers
        RF[t] = C[t] *p_f[t] -c_t *E[t]
        # revenue of traders
        RT[t] = C[t] *p_e[t] -RF[t] -c_p
        # revenue of all fishery
        RA[t] = C[t] *p_e[t]

        # pay gap
        G[t] = RF[t]/RT[t]

    return tau, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA, G

################################################################################
#############################  RUN MODEL FILE  #################################
################################################################################

##### Run the model ############################################################
OUT = np.zeros(tau.shape[0])
OUT1 = np.zeros(tau.shape[0])
OUT2 = np.zeros(tau.shape[0])

for i in np.arange(0,tau.shape[0]):
        tau, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA, G = model(competition, demand, d1, d2, b0, b1, b2, b3, l1, l2, qc, a1, d, f, g, K, h1, h2, gamma, beta, c_p, c_t, flag)
        OUT[i]= RF[i]
        OUT1[i]= p_f[i]
        OUT2[i]= p_e[i]

##### Save data  ###############################################################
# if intervention == 0:
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_NoInt_RF.npy", OUT)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_NoInt_RT.npy", OUT1)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_NoInt_RA.npy", OUT2)
# if intervention == 1:
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntD_RF.npy", OUT)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntD_RT.npy", OUT1)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntD_RA.npy", OUT2)
# if intervention == 2:
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntC_RF.npy", OUT)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntC_RT.npy", OUT1)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntC_RA.npy", OUT2)


################################################################################
###############################  PLOT FILE  ####################################
################################################################################

##### Load data  ###############################################################
# no intervention
noF= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_NoInt_RF.npy") # Revenue fisher
noT= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_NoInt_RT.npy") # Revenue trader
noA= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_NoInt_RA.npy") # Revenue all fishery
# intervention demand
dF= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntD_RF.npy")
dT= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntD_RT.npy")
dA= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntD_RA.npy")
# intervention competition
cF= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntC_RF.npy")
cT= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntC_RT.npy")
cA= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3support1_IntC_RA.npy")



#####! PLOT MODEL  #############################################################
##! set font
hfont = {'fontname':'Helvetica'}

# begin plotting demand intervention
fig = plt.figure()
a, = plt.plot(dF, label = "Fishers income", color = 'steelblue')
b, = plt.plot(dT, label = "Traders income", color = 'orange')
c, = plt.plot(dA, label = "Fishery revenue", color = 'indianred')
# x-axis
#plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(2,tmax)
plt.xlabel("time $years$",fontsize=22, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
plt.ylim(0,4E9)
plt.ylabel("Value $MXN$" ,fontsize=22, **hfont)
plt.legend(handles=[a,b,c], loc='best', fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R4_support1.png',dpi=500)
plt.show()


# begin plotting competition intervention
fig = plt.figure()
a, = plt.plot(cF, label = "Fishers income", color = 'steelblue')
b, = plt.plot(cT, label = "Traders income", color = 'orange')
c, = plt.plot(cA, label = "Fishery revenue", color = 'indianred')
# x-axis
#plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(2,tmax)
plt.xlabel("time $years$",fontsize=22, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
plt.ylim(0,4E9)
plt.ylabel("Value $MXN$" ,fontsize=22, **hfont)
plt.legend(handles=[a,b,c], loc='best', fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R4_support2.png',dpi=500)
plt.show()


################################################################################
###############################  DIRECT PLOTS  #################################
################################################################################

##! set font
hfont = {'fontname':'Helvetica'}

# begin plotting demand intervention
fig = plt.figure()
a, = plt.plot(OUT1, label = "Fishers price", color = 'steelblue')
b, = plt.plot(OUT2, label = "Traders price", color = 'orange')
# x-axis
#plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(2,tmax)
plt.xlabel("time $years$",fontsize=22, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
# plt.ylim(0,4E9)
plt.ylabel("Value $MXN$" ,fontsize=22, **hfont)
plt.legend(handles=[a,b,c], loc='best', fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R4_support1.png',dpi=500)
plt.show()
