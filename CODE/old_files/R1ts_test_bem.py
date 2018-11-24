#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy import stats
from pandas import *

#### Model w/o relationships ###################################################
flag = 0 # 0 = NoR model; 1 = Rmodel

### Parameters #################################################################
# scales: tons, years, MXNia, hours, trips
tmax = 100 # model run, years

# following parameters fitted to SST
b0 = -16.49 # SST trend
b1 = 0.02 # SST trend
b2 = 6.779 # SST trend
b3 = 0.091 # SST trend
n1 = -22.239 # ML, slope
n2 = 49.811 # ML, intersect
l1 = -0.0059 # q, slope
l2 = 0.1882 # q, intersect
qc = 0.1 # q, constant catchability
a1 = 1/np.exp(32-(b0+b1*2015)) # proportion of migrating squid, where 3.4E7 max(e^(tau-b1))
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
# tmax = len(y)

### Define Model ###############################################################
def model(b0, b1, b2, b3, n1, n2, l1, l2, qc, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag):
    for t in np.arange(1,tmax):
        # sst trend
        tau[t]= b0 +b1 *(t+2015) +b2 *np.cos(t+2015) + b3 *np.sin(t+2015)
        # isotherm depth
        # tau[t]= b1 +b2 *np.cos(t) + b3 *np.sin(t)

        # catchability sst
        q[t]= l1 *tau[t] +l2
        if q[t] > 0.1:
            q[t] = 0.1
            print "q high"
        elif q[t] < 0:
            q[t] = 0
            print "q low"
        # mantle length and catchability
        # if ml[t] == 1:
        #     q[t]= l1 *tau[t] +l2
        # else:
        #     ML[t]= ml[t]
        #     q[t]= 0.0018 *ML[t] - 0.0318

        # migration of squid
        y_S[t] = a1 *np.exp(tau[t]-(b0 +b1*(t+2015))) # sst trend
        if y_S[t] > 1:
            y_S[t] = 1
            print "yS high"
        elif y_S[t] < 0.01:
            y_S[t] = 0.01
            print "yS low"

        # trader cooperation
        R_tt[t] = (1-y_S[t])

        #### switch between models ####
        if flag == 0: # squid population BEM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - qc *E[t-1] *S[t-1]
        if flag == 1: # squid population MLM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        # cost of transport
        c_t[t]= m *f # I decided to use fixed costs over migration, that equally well/better predicted catches over m* (y_S[t]); (source: LabNotesSquid, April 11)

        #### switch between models ####
        if flag == 0: # effort BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -c_t[t-1] *(E[t-1]/(B_h*B_f)) # c_t is per trip so we need to upscale E hr > fisher > trip
        if flag == 1: # effort MLM
            Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -c_t[t-1] *(E[t-1]/(B_h*B_f)) # c_t is per trip so we need to upscale E hr > fisher > trip

        # fishing effort
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
        if flag == 1: # catch MLM
            C[t] = q[t] *E[t] *S[t]

        # export price
        p_e[t] = gamma* (C[t])**(-beta)
        if p_e[t]>= 99366:
            p_e[t]= 99366
            print "pe high"

        #### switch between models ####
        if flag == 0:
            # price for fishers
            p_f[t] = p_e[t] -c_p
            print "BEM"
        if flag == 1:
            # minimum wage
            p_min[t]= c_t[t] *(E[t]/(B_h*B_f))
            # price for fishers
            p_f[t] = (p_e[t] -c_p) *(1-R_tt[t]) +R_tt[t] *p_min[t]
            print "MLM"

        print q[t], y_S[t], S[t], E[t], C[t], p_e[t], p_f[t]
    return tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f

################################################################################
###########################  RUN MODEL FILE  ###################################
################################################################################

##### Initiate arrays ##########################################################
sim = np.arange(0,1) # number of simulations
x = np.zeros(10) # set array to save parameters
par = np.zeros((sim.shape[0],x.shape[0])) # matrix to save parameter values of each simulation
cat = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save catches in each time period of each simulation
pri = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save prices for fishers in each time period of each simulation

### extra variables to monitor
tem = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save tau in each time period of each simulation
mig = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save migrate squid in each time period of each simulation
cco = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save catchability in each time period of each simulation
pop = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save squid population in each time period of each simulation
eff = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save effort in each time period of each simulation
mar = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save market prices in each time period of each simulation

##### Run the model ############################################################
for j in range(0,sim.shape[0]): # draw randomly a float in the range of values for each parameter
    # mute the random selection to choose your parameter values
    # b0 = np.random.uniform(-5.15, -27.83)
    # b1 = np.random.uniform(0.026, 0.014)
    # b2 = np.random.uniform(6.859, 6.699)
    # b3 = np.random.uniform(0.171, 0.011)
    # beta = np.random.uniform(0.01, 0.1)
    # c_p = np.random.uniform(1000, 2148)
    # g = np.random.uniform(0, 2.9)
    # gamma = np.random.uniform(2000, 51000)
    # m = np.random.uniform(2368793, 8450159)
    # w_m = np.random.uniform(58528541, 156076110)

    x = [b0, b1, b2, b3, beta, c_p, g, gamma, m , w_m]
    par[j] = x
    print par

    OUT1 = np.zeros(tau.shape[0])
    OUT2 = np.zeros(tau.shape[0])
    OUT3 = np.zeros(tau.shape[0])
    OUT4 = np.zeros(tau.shape[0])
    OUT5 = np.zeros(tau.shape[0])
    OUT6 = np.zeros(tau.shape[0])
    OUT7 = np.zeros(tau.shape[0])
    OUT8 = np.zeros(tau.shape[0])

    for i in np.arange(0,1):
            tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f = model(b0, b1, b2, b3, n1, n2, l1, l2, qc, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag)
            OUT1[i]= p_f[i]
            OUT2[i]= C[i]
            OUT3[i]= tau[i]
            OUT4[i]= y_S[i]
            OUT5[i]= q[i]
            OUT6[i]= S[i]
            OUT7[i]= E[i]
            OUT8[i]= p_e[i]
            print "END"
            print "b0", b0, b0 -(-16.49)
            print "b1", b1, b1 -(0.02)
            print "b2", b2, b2 -(6.779)
            print "b3", b3, b3 -(0.091)
            print "beta", beta, beta-(0.0736)
            print "c_p", c_p, c_p -(1776.25)
            print "g", g, g -(1.4)
            print "gamma", gamma, gamma -(49200)
            print "m", m , m -(156076110)
            print "w_m", w_m, w_m -(13355164)

# activate this area only as you have 1 < MC runs
            # pri[j,i] = OUT1[i]
            # cat[j,i] = OUT2[i]
            # tem[j,i] = OUT3[i]
            # mig[j,i] = OUT4[i]
            # cco[j,i] = OUT5[i]
            # pop[j,i] = OUT6[i]
            # eff[j,i] = OUT7[i]
            # mar[j,i] = OUT8[i]

# lowC = np.zeros(y.shape[0]) # initiate variables for 95% confidence interval
# highC = np.zeros(y.shape[0])
# meanC = np.zeros(y.shape[0])
# lowP = np.zeros(y.shape[0])
# highP = np.zeros(y.shape[0])
# meanP = np.zeros(y.shape[0])
#
# for h in range(0,y.shape[0]): # calculate the 95% confidence interval
#     z = cat[:,h]
#     lowC[h] = np.nanmean(z) - ((1.96 * np.nanstd(z))/np.sqrt(np.count_nonzero(~np.isnan(z))))
#     highC[h] = np.nanmean(z) + ((1.96 * np.nanstd(z))/np.sqrt(np.count_nonzero(~np.isnan(z))))
#     meanC[h] = np.nanmean(z)
#     zeta = pri[:,h]
#     lowP[h] = np.nanmean(zeta) - ((1.96 * np.nanstd(zeta))/np.sqrt(np.count_nonzero(~np.isnan(zeta))))
#     highP[h] = np.nanmean(zeta) + ((1.96 * np.nanstd(zeta))/np.sqrt(np.count_nonzero(~np.isnan(zeta))))
#     meanP[h] = np.nanmean(zeta)

################################################################################
###########################  PLOT FILE  ########################################
################################################################################

### New max time ###############################################################
yr = np.arange(0,27)
tmax = len(yr)
x = np.arange(0,len(yr))

### font ######################################################################
hfont = {'fontname':'Helvetica'}

#####! PLOT MODEL  #############################################################
# Three subplots sharing both x/y axes
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# axis 1
ax1.plot(tau, label = "tau", color="orange")
ax1.set_title('Temperature')
ax1.legend(loc="best")
# axis 2
ax2.plot(y_S, label = "migration", color="orange")
ax2.plot(q, label = "catchability", color="steelblue")
ax2.plot(E, label = "effort", color="red")
ax2.legend(loc="best")
# axis 3
ax3.plot(C, label = "catch", color="orange")
ax3.plot(S, label = "population", color="steelblue")
ax3.legend(loc="best")
# axis 4
ax4.plot(p_f, label = "price for fishers", color="orange")
ax4.plot(p_e, label = "market price", color="steelblue")
ax4.legend(loc="best")
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0.2)
plt.show()

#
# # plot tau
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# a, = plt.plot(OUT3, label = "tau", color="orange")
# # x-axis
# plt.xticks(np.arange(len(yr)), yr, rotation=45, fontsize=12)
# # plt.xlim(10,tmax-2)
# plt.xlabel("Year",fontsize=22, **hfont)
# plt.gcf().subplots_adjust(bottom=0.15)
# # y-axis
# plt.ylabel("Temperature",fontsize=22, **hfont)
# # legend
# plt.legend(handles=[a], loc='best', fontsize=14)
# # save and show
# plt.show()
#
# # plot y_S, q, E
# fig = plt.figure()
# a, = plt.plot(OUT4, label = "migration", color="orange")
# b, = plt.plot(OUT5, label = "catchability", color="steelblue")
# c, = plt.plot(OUT7, label = "effort", color="red")
# # x-axis
# plt.xticks(np.arange(len(yr)), yr, rotation=45, fontsize=12)
# # plt.xlim(10,tmax-2)
# plt.xlabel("Year",fontsize=22, **hfont)
# plt.gcf().subplots_adjust(bottom=0.15)
# # y-axis
# plt.ylabel("See legend",fontsize=22, **hfont)
# # legend
# plt.legend(handles=[a,b,c], loc='best', fontsize=14)
# # save and show
# plt.show()
#
# # plot S, C
# fig = plt.figure()
# a, = plt.plot(OUT2, label = "catch", color="orange")
# b, = plt.plot(OUT6, label = "population", color="steelblue")
# # x-axis
# plt.xticks(np.arange(len(yr)), yr, rotation=45, fontsize=12)
# # plt.xlim(10,tmax-2)
# plt.xlabel("Year",fontsize=22, **hfont)
# plt.gcf().subplots_adjust(bottom=0.15)
# # y-axis
# plt.ylabel("volume $tons$",fontsize=22, **hfont)
# # legend
# plt.legend(handles=[a,b], loc='best', fontsize=14)
# # save and show
# plt.show()
#
# # plot pf, pe
# fig = plt.figure()
# a, = plt.plot(OUT1, label = "price for fishers", color="orange")
# b, = plt.plot(OUT8, label = "market price", color="steelblue")
# # x-axis
# plt.xticks(np.arange(len(yr)), yr, rotation=45, fontsize=12)
# # plt.xlim(10,tmax-2)
# plt.xlabel("Year",fontsize=22, **hfont)
# plt.gcf().subplots_adjust(bottom=0.15)
# # y-axis
# plt.ylabel("price $MXN$",fontsize=22, **hfont)
# # legend
# plt.legend(handles=[a,b], loc='best', fontsize=14)
# # save and show
# plt.show()
#
