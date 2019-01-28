#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy
from pandas import *

#################### CHOICES ###################################################
flag = 1 # 0 = BEM; 1 = MLM, # 2 = BLM
# SST or SSTres: parameters, initial condition for tau, and MC parameter ranges
mantle = 1 ### default 1 for this model # 0 = use mantle, 1 = use tau
intervention = 0 # 0 = competition intervention; 1 = demand BEM; 2 = demand BLM; 3= demand MLM
competition = 1 # 0 = no intervention; 1 = intervention competition
demand = 0 # 0 = no intervention; 1 = intervention demand
migrate = 1 # 0 = use discrete function, 1 = use continuous function, 2 = use data

# for competition int: 110101
# for demand BEM: 011011
# for demand BLM: 212011
# for demand MLM: 113011

### Parameters #################################################################
tmax = 35 # model run, years
b0 = -40.901 #
b1 = 0.020 #
b2 = 0.165 #
b3 = -0.287 #
l1 = -0.0912  #
l2 = 0.0231 #
c_t = 107291548 # cost of fishing
qc = 0.1 # catchability constant
f = 0 # intercept of trader cooperation
d = 1 # slope of trader cooperation
K = 1208770 # carrying capacity
g = 1.4 # population growth rate
gamma = 49200 # maximum demand
beta = 0.0736 # slope of demand-price function
c_p = 1776.25 # cost of processing

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
p_min = np.zeros(tmax) # minimum wage
p_f = np.zeros(tmax) # price for fishers
RFn = np.zeros(tmax) # revenue of fishers normalized
RF = np.zeros(tmax) # revenue of fishers
RT = np.zeros(tmax) # revenue of traders
RA = np.zeros(tmax) # revenue all fishery
G = np.zeros(tmax) # pay gap between fishers and traders

### Initial values #############################################################
tau[0] = -0.80 # for SSTres
q[0] = 0.05 # squid catchability
y_S[0] = 0.5 # proportion of migrated squid
R_tt[0] = 0.5 # trader cooperation
S[0] = 610075 # size of the squid population, mean
E[0] = 0.5 # fishing effort
C[0] = 60438 # squid catch mean
p_e[0] = 52035 # mean p_e comtrade, rounded
p_f[0] = 8997 # mean p_f datamares, rounded

### intervention parameters and variables ######################################
### parameters
i_e = 0.1 # increment of decrease in traders cooperation per unit investment
timeInt = 15 # year to start intervention

### Variables
F = np.zeros(tmax) # intercept of traders cooperation

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

### continuous migration trigger ###############################################
## calculates migration from a continuous sin/cos function
xo = np.linspace(1991,2025,1000) # 100 linearly spaced numbers
#y = np.sin(x)/x # computing the values of sin(x)/x
yo = 4.1E-15 *np.exp(100*(b2*np.cos(xo)-b3*np.sin(xo)))
plt.plot(xo,yo) # sin(x)/x
plt.show()

yo = np.array([0,0,0.9773642085169757,0,0,0,0,0,0.9773642085169757,0,0,0,0,0,0.9773642085169757,0,0,0,0,0,0.9773642085169757,0,0,0,0,0,0, 0.9773642085169757,0,0,0,0,0,0.9773642085169757,0])

### Normalizes q values ########################################################
## calculates q as normalized value of tau over the time span provided
qMax = 0 # is reverse bc is reverse to tau values
qMin = 0.2
valueScaled = np.zeros(tmax)

def translate(b0, b1, b2, b3, tau, qMin, qMax):
    tau[t]= b0 +b1 *(t+1990) +b2 *np.cos(t+1990) + b3 *np.sin(t+1990)

    # Figure out how 'wide' each range is
    tauSpan = max(tau) - min(tau)
    qSpan = qMax - qMin

    # Convert the left range into a 0-1 range (float)
    valueScaled[t] = float(tau[t] - min(tau)) / float(tauSpan)

    # Convert the 0-1 range into a value in the right range.
    q = qMin + (valueScaled * qSpan)

    return q, tauSpan, qSpan

for t in np.arange(0,tmax):
     q, tauSpan, qSpan = translate(b0, b1, b2, b3, tau, qMin, qMax)

tauMin = min(tau)


################################################################################
###########################  MODEL FILE  #######################################
################################################################################

### Define Model ###############################################################
def model(b0, b1, b2, b3, l1, l2, qc, qMin, qSpan, tauSpan, tauMin, d, f, g, K, h1, h2, gamma, beta, c_p, c_t, flag):
    for t in np.arange(1,tmax):
        tau[t]= b0 +b1 *(t+1990) +b2 *np.cos(t+1990) + b3 *np.sin(t+1990)

        # mantle length and catchability
        if mantle == 0:
            if ml[t] == 1:
                valueScaled[t] = float(tau[t] - tauMin) / float(tauSpan)
                q[t] = qMin + (valueScaled[t] * qSpan)
            else:
                ML[t]= ml[t]
                q[t]= 0.0018 *ML[t] - 0.0318
        if mantle == 1:
            valueScaled[t] = float(tau[t] - tauMin) / float(tauSpan)
            q[t] = qMin + (valueScaled[t] * qSpan)

        if q[t] > qc:
            q[t] = qc
            print "q high"
        elif q[t] < 0:
            q[t] = 0
            print "q low"

        # migration of squid
        if migrate == 0: # use discrete function
            y_S[t] = 4.1E-15 *np.exp(100*(b2*np.cos(t)-b3*np.sin(t)))

        if migrate == 1: # use continuous function
                y_S[t]= yo[t] # run with continuous function

        if migrate == 2: # use data input: data is not available for the entire timeseries, the data gaps are filled with continuous function simulations
            if ys[t] == 1:
                y_S[t]= yo[t]
            else:
                y_S[t]= ys[t]

        if y_S[t] > 1:
            y_S[t] = 1
            print "yS high"
        elif y_S[t] < 0:
            y_S[t] = 0
            print "yS low"

        ## trader cooperation intervention
        if t <= timeInt:
            R_tt[t] = f+ np.exp(-d* y_S[t])
        else:
            if competition == 0:
                R_tt[t] = f+ np.exp(-d* y_S[t])
            if competition == 1:
                F[t] = F[t-1]- (i_e * R_tt[t-1])
                R_tt[t] = F[t]+ np.exp(-d* y_S[t])

        #### switch between models ####
        if flag == 0: # squid population BEM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - qc *E[t-1] *S[t-1]
        if flag >= 1: # squid population MLM, BLM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        #### switch between models ####
        if flag == 0: # effort BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -c_t *E[t-1] # c_t is per trip so we need to upscale E hr > fisher > trip
            E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]

        if flag >= 1: # effort MLM, BLM
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
        if flag >= 1: # catch MLM, BLM
            C[t] = q[t] *E[t] *S[t]

        if C[t] <= 0: # avoid infinity calculations
            C[t]= 1

        ## demand intervention
        if t <= timeInt:
            p_e[t] = gamma* (C[t])**(-beta)
        else:
            if demand == 0:
                p_e[t] = gamma* (C[t])**(-beta)
            if demand == 1:
                p_e[t] = (gamma *(1+ i_e*0.4 *(t -timeInt))) *(C[t])**(-beta)

        # export price
        if p_e[t] >= 99366:
            p_e[t] = 99366
            print "pe high"

        #### switch between models ####
        if flag == 0: # BEM
            # price for fishers
            p_f[t] = p_e[t] -c_p
        if flag == 2: # BLM
            p_f[t] = p_e[t] -c_p    # price for fishers
        if flag == 1: #MLM
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

        # print t, tau[t], ML[t], q[t], y_S[t], S[t], E[t], C[t], p_e[t], p_f[t]
    return tau, ML, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA, G

################################################################################
###########################  RUN MODEL FILE  ###################################
################################################################################

##### initiate output ###########################################################
OUT1 = np.zeros(tau.shape[0])
OUT2 = np.zeros(tau.shape[0])
OUT3 = np.zeros(tau.shape[0])
OUT4 = np.zeros(tau.shape[0])
OUT5 = np.zeros(tau.shape[0])
OUT6 = np.zeros(tau.shape[0])
OUT7 = np.zeros(tau.shape[0])
OUT8 = np.zeros(tau.shape[0])
OUT9 = np.zeros(tau.shape[0])
OUT10 = np.zeros(tau.shape[0])
OUT11 = np.zeros(tau.shape[0])
OUT12 = np.zeros(tau.shape[0])

##### Run the model ############################################################
for i in np.arange(1,tmax):
        tau, ML, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA, G = model(b0, b1, b2, b3, l1, l2, qc, qMin, qSpan, tauSpan, tauMin, d, f, g, K, h1, h2, gamma, beta, c_p, c_t, flag)
        OUT1[i]= p_f[i]
        OUT2[i]= C[i]
        OUT3[i]= tau[i]
        OUT4[i]= y_S[i]
        OUT5[i]= q[i]
        OUT6[i]= S[i]
        OUT7[i]= E[i]
        OUT8[i]= p_e[i]
        OUT9[i]= G[i]
        OUT10[i]= RF[i]
        OUT11[i]= RT[i]
        OUT12[i]= RA[i]

################################################################################
###########################  PLOT FILE FROM OUTPUT  ############################
################################################################################
hfont = {'fontname':'Helvetica'}

### simple plot
# begin plotting demand intervention
fig = plt.figure()
a, = plt.plot(OUT10, label = "Income fishers", color = 'steelblue')
b, = plt.plot(OUT11, label = "Income traders", color = 'orange')
c, = plt.plot(OUT12, label = "Revenue fishery", color = 'sage')
# x-axis
#plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(1,tmax)
plt.xlabel("time $years$",fontsize=22, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
# plt.ylim(0,4E9)
plt.ylabel("Income and revenue $MXN$" ,fontsize=22, **hfont)
plt.legend(handles=[a,b,c], loc='best', fontsize=14)
# load and show
plt.show()

### advanced plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
a, = ax1.plot(OUT10, label = "Fishers income", color = 'steelblue')
b, = ax1.plot(OUT11, label = "Traders income", color = 'orange')
c, = ax1.plot(OUT12, label = "Revenue Fishery", color = 'indianred')
# x-axis
# add the second axes using subplot with ML
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
d, = ax2.plot(OUT3, color="lightgrey")
# x-axis
plt.xlim(2,tmax)
plt.xlabel("time $years$",fontsize=22, **hfont)
ax1.set_xticklabels(np.arange(1990,2025,5), rotation=45, fontsize= 12)
ax2.set_xticklabels(np.arange(1990,2025,5), rotation=45, fontsize= 12)
# y-axis
ax1.set_ylabel("Revenue $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax1.set_ylim(0,3E9)
ax2.set_ylabel("SST anomaly $C$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='lightgrey')
ax2.set_ylim(-6,0)
# adjusting labels and plot size
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(handles=[a,b,c,d], loc='best', fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R4_support2.png',dpi=500)
plt.show()


##### Save data  ###############################################################
if intervention == 0: # intervention competition
    print "intervention competition"
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_pf.npy", OUT1)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_pe.npy", OUT8)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_RF.npy", OUT10)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_RT.npy", OUT11)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_RA.npy", OUT12)
if intervention == 1: # intervention demand BEM
    print "intervention demand BEM"
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_pf.npy", OUT1)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_pe.npy", OUT8)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_RF.npy", OUT10)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_RT.npy", OUT11)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_RA.npy", OUT12)
if intervention == 2: # intervention demand BLM
    print "intervention demand BLM"
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_pf.npy", OUT1)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_pe.npy", OUT8)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_RF.npy", OUT10)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_RT.npy", OUT11)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_RA.npy", OUT12)
if intervention == 3: # intervention demand MLM
    print "intervention demand MLM"
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_pf.npy", OUT1)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_pe.npy", OUT8)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_RF.npy", OUT10)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_RT.npy", OUT11)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_RA.npy", OUT12)

# ################################################################################
# ###########################  PLOT FILE  ########################################
# ################################################################################
#
# ##### Load data  ###############################################################
# competition intervention
cPE = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_pe.npy")
cPF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_pf.npy")
cRF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_RF.npy")
cRT = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_RT.npy")
cRA = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_Rtt_RA.npy")
# intervention demand BEM
dBPE = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_pe.npy")
dBPF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_pf.npy")
dBRF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_RF.npy")
dBRT = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_RT.npy")
dBRA = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BEM_RA.npy")
# intervention demand BLM
dLPE = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_pe.npy")
dLPF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_pf.npy")
dLRF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_RF.npy")
dLRT = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_RT.npy")
dLRA = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_BLM_RA.npy")
# intervention demand MLM
dMPE = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_pe.npy")
dMPF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_pf.npy")
dMRF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_RF.npy")
dMRT = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_RT.npy")
dMRA = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R4_gamma_MLM_RA.npy")
#
#
### font ######################################################################
hfont = {'fontname':'Helvetica'}

yr = np.arange(1990,2025,1)

# begin plotting demand intervention
fig = plt.figure()
ax1 = fig.add_subplot(111)
a, = ax1.plot(dMRF, label = "Fisher, MLM", color = 'steelblue', linestyle='-')
b, = ax1.plot(dMRT, label = "Trader, MLM", color = 'orange', linestyle='-')
c, = ax1.plot(dLRF, label = "EDM", color = 'steelblue', linestyle='--')
d, = ax1.plot(dLRT, label = "Trader, EDM", color = 'orange', linestyle='--')
e, = ax1.plot(dBRF, label = "BEM", color = 'steelblue', linestyle=':')
f, = ax1.plot(dBRT, label = "Trader, BEM", color = 'orange', linestyle=':')
# x-axis
# add the second axes using subplot with ML
# ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
# g, = ax2.plot(OUT3, color="lightgrey")
# x-axis
plt.xlim(2,tmax)
plt.xlabel("Year",fontsize=22, **hfont)
ax1.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 12)
# ax2.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 12)
# y-axis
ax1.set_ylabel("Income $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax1.set_ylim(-.5E9,4E9)
# ax2.set_ylabel("SST anomaly $C$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
# ax2.tick_params(axis='y', colors='lightgrey')
# ax2.set_ylim(-2,10)
# adjusting labels and plot size
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(handles=[a,b,c,e], loc=2, fontsize=12)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R4_support1.png',dpi=500)
plt.show()


# begin plotting competition intervention
fig = plt.figure()
ax1 = fig.add_subplot(111)
a, = ax1.plot(cRF, label = "Fisher, MLM", color = 'steelblue')
b, = ax1.plot(cRT, label = "Trader, MLM", color = 'orange')
# x-axis
# add the second axes using subplot with ML
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
c, = ax2.plot(OUT3, color="lightgrey")
# x-axis
plt.xlim(2,tmax)
plt.xlabel("Year",fontsize=22, **hfont)
ax1.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 12)
ax2.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 12)
# y-axis
ax1.set_ylabel("Income $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax1.set_ylim(-0.5E9,2.5E9)
ax2.set_ylabel("SST anomaly $C$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='lightgrey')
ax2.set_ylim(-1.5,5.5)
# adjusting labels and plot size
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(handles=[a,b], loc='best', fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R4_support2.png',dpi=500)
plt.show()
