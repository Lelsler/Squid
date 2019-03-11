#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import *
import scipy

#################### CHOICES ###################################################
flag = 1 # 0 = BEM; 1 = MLM, # 2 = BLM
# SST or SSTres: parameters, initial condition for T, and MC parameter ranges
mantle = 1 ### default 1 for this model # 0 = use mantle, 1 = use T
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
a0 = -40.910 #
a1 = 0.020 #
a2 = 0.165 #
a3 = -0.287 #
k = -0.0912  #
l = 0.0231 #
sigma = 107291548 # cost of fishing
qc = 0.1 # catchability constant
f = 0 # intercept of trader cooperation
delta = 1 # slope of trader cooperation
K = 1208770 # carrying capacity
g = 1.4 # population growth rate
gamma = 49200 # maximum demand
beta = 0.0736 # slope of demand-price function
kappa = 1776.25 # cost of processing

h1 = 2E-10 # scale E
h2 = 0.6596 # scale E

### Variables ##################################################################
T = np.zeros(tmax) # temperature
q = np.zeros(tmax) # catchability squid population
ML = np.zeros(tmax) # mantle length
M = np.zeros(tmax) # distance of squid migration from initial fishing grounds
R = np.zeros(tmax) # trader cooperation
S = np.zeros(tmax) # size of the squid population
Escal = np.zeros(tmax) # scale effort
E = np.zeros(tmax) # fishing effort
C = np.zeros(tmax) # squid catch
p_m = np.zeros(tmax) # export price
p_min = np.zeros(tmax) # minimum wage
p_f = np.zeros(tmax) # price for fishers
I_fn = np.zeros(tmax) # revenue of fishers normalized
I_f = np.zeros(tmax) # revenue of fishers
I_t = np.zeros(tmax) # revenue of traders
RA = np.zeros(tmax) # revenue all fishery
G = np.zeros(tmax) # pay gap between fishers and traders

### Initial values #############################################################
T[0] = -0.80 # for SSTres
q[0] = 0.05 # squid catchability
M[0] = 0.5 # proportion of migrated squid
R[0] = 0.5 # trader cooperation
S[0] = 610075 # size of the squid population, mean
E[0] = 0.5 # fishing effort
C[0] = 60438 # squid catch mean
p_m[0] = 52035 # mean p_m comtrade, rounded
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
ct = df1['sigma'] #
ssh = df1['essh_avg'] #
ml = df1['ML'] #
ys = df1['M'] #

### continuous migration trigger ###############################################
## calculates migration from a continuous sin/cos function
xo = np.linspace(1980,1990,1000) # 1000 linearly spaced numbers
#temp = a0 +a1 *(xo) +a2 *np.cos(xo) + a3 *np.sin(xo)
temp = -40.910 + 0.165*np.cos(xo)-0.287*np.sin(xo)+0.02*xo
yo = alpha *np.exp(100*(a2*np.cos(xo)-a3*np.sin(xo)))
plt.plot(xo,yo) # sin(x)/x
plt.show()

y = np.linspace(1991,2025,1000)
x = np.zeros(1000)
for i in np.arange(0,1000,1):
    y[i] = alpha *np.exp(100*(a2*np.cos(xo[i])-a3*np.sin(xo[i])))
    if y[i] > 0.9:
         x[i] = xo[i]

x = np.around(x, decimals=0)
plt.plot(xo,x) # sin(x)/x


### Normalizes q values ########################################################
## calculates q as normalized value of T over the time span provided
qMax = 0 # is reverse bc is reverse to T values
qMin = 0.1
valueScaled = np.zeros(tmax)

def translate(a0, a1, a2, a3, T, qMin, qMax):
    T[t]= a0 +a1 *(t+1990) +a2 *np.cos(t+1990) + a3 *np.sin(t+1990)

    # Figure out how 'wide' each range is
    TSpan = max(T) - min(T)
    qSpan = qMax - qMin

    # Convert the left range into a 0-1 range (float)
    valueScaled[t] = float(T[t] - min(T)) / float(TSpan)

    # Convert the 0-1 range into a value in the right range.
    q = qMin + (valueScaled * qSpan)

    return q, TSpan, qSpan

for t in np.arange(0,tmax):
     q, TSpan, qSpan = translate(a0, a1, a2, a3, T, qMin, qMax)

TMin = min(T)


################################################################################
###########################  MODEL FILE  #######################################
################################################################################

### Define Model ###############################################################
def model(a0, a1, a2, a3, k, l, qc, qMin, qSpan, TSpan, TMin, alpha, delta, g, K, h1, h2, gamma, beta, kappa, sigma, flag):
    for t in np.arange(1,tmax):
        T[t]= a0 +a1 *(t+1990) +a2 *np.cos(t+1990) + a3 *np.sin(t+1990)
        time = 1990 +t

        # mantle length and catchability
        if mantle == 0:
            if ml[t] == 1:
                valueScaled[t] = float(T[t] - TMin) / float(TSpan)
                q[t] = qMin + (valueScaled[t] * qSpan)
            else:
                ML[t]= ml[t]
                q[t]= 0.0018 *ML[t] - 0.0318
        if mantle == 1:
            valueScaled[t] = float(T[t] - TMin) / float(TSpan)
            q[t] = qMin + (valueScaled[t] * qSpan)

        if q[t] > qc:
            q[t] = qc
            print "q high"
        elif q[t] < 0:
            q[t] = 0
            print "q low"

        # migration of squid
        if migrate == 0: # use discrete function
            M[t] = alpha *np.exp(100*(a2*np.cos(t)-a3*np.sin(t)))

        if migrate == 1: # use continuous function
                if any(time == x):
                    M[t] = 1
                #M[t]= yo[t] # run with continuous function

        if migrate == 2: # use data input: data is not available for the entire timeseries, the data gaps are filled with continuous function simulations
            if ys[t] == 1:
                M[t]= yo[t]
            else:
                M[t]= ys[t]

        if M[t] > 1:
            M[t] = 1
            print "yS high"
        elif M[t] < 0:
            M[t] = 0
            print "yS low"

        ## trader cooperation intervention
        if t <= timeInt:
            R[t] = np.exp(-delta* M[t])
        else:
            if competition == 0:
                R[t] = np.exp(-delta* M[t])
            if competition == 1:
                F[t] = F[t-1]- (i_e * R[t-1])
                R[t] = F[t]+ np.exp(-delta* M[t])

        #### switch between models ####
        if flag == 0: # squid population BEM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - qc *E[t-1] *S[t-1]
        if flag >= 1: # squid population MLM, BLM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        #### switch between models ####
        if flag == 0: # effort BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -sigma *E[t-1] # sigma is per trip so we need to upscale E hr > fisher > trip
            E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]

        if flag >= 1: # effort MLM, BLM
            Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -sigma *E[t-1] # sigma is per trip so we need to upscale E hr > fisher > trip
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
            p_m[t] = gamma* (C[t])**(-beta)
        else:
            if demand == 0:
                p_m[t] = gamma* (C[t])**(-beta)
            if demand == 1:
                p_m[t] = (gamma *(1+ i_e*0.4 *(t -timeInt))) *(C[t])**(-beta)

        # export price
        if p_m[t] >= 99366:
            p_m[t] = 99366
            print "pe high"

        #### switch between models ####
        if flag == 0: # BEM
            # price for fishers
            p_f[t] = p_m[t] -kappa
        if flag == 2: # BLM
            p_f[t] = p_m[t] -kappa    # price for fishers
        if flag == 1: #MLM
            p_min[t]= (sigma *E[t])/C[t] # ->must be MXN/ton to fit into p_f calc
            # price for fishers
            p_f[t] = (p_m[t] -kappa) *(1-R[t]) +R[t] *p_min[t]

        if p_f[t] >= (p_m[t] -kappa): # limit price of fishers
            p_f[t] = (p_m[t] -kappa)

        # revenue of fishers
        I_f[t] = C[t] *p_f[t] -sigma *E[t]
        # revenue of traders
        I_t[t] = C[t] *p_m[t] -I_f[t] -kappa
        # revenue of all fishery
        RA[t] = C[t] *p_m[t]

        # pay gap
        G[t] = I_f[t]/I_t[t]

        # print t, T[t], ML[t], q[t], M[t], S[t], E[t], C[t], p_m[t], p_f[t]
    return T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, RA, G

################################################################################
###########################  RUN MODEL FILE  ###################################
################################################################################

##### initiate output ###########################################################
OUT1 = np.zeros(T.shape[0])
OUT2 = np.zeros(T.shape[0])
OUT3 = np.zeros(T.shape[0])
OUT4 = np.zeros(T.shape[0])
OUT5 = np.zeros(T.shape[0])
OUT6 = np.zeros(T.shape[0])
OUT7 = np.zeros(T.shape[0])
OUT8 = np.zeros(T.shape[0])
OUT9 = np.zeros(T.shape[0])
OUT10 = np.zeros(T.shape[0])
OUT11 = np.zeros(T.shape[0])
OUT12 = np.zeros(T.shape[0])

##### Run the model ############################################################
for i in np.arange(1,tmax):
        T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, RA, G = model(a0, a1, a2, a3, k, l, qc, qMin, qSpan, TSpan, TMin, alpha, delta, g, K, h1, h2, gamma, beta, kappa, sigma, flag)
        OUT1[i]= p_f[i]
        OUT2[i]= C[i]
        OUT3[i]= T[i]
        OUT4[i]= M[i]
        OUT5[i]= q[i]
        OUT6[i]= S[i]
        OUT7[i]= E[i]
        OUT8[i]= p_m[i]
        OUT9[i]= G[i]
        OUT10[i]= I_f[i]
        OUT11[i]= I_t[i]

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
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
g, = ax2.plot(OUT3, color="lightgrey")
# x-axis
plt.xlim(2,tmax)
plt.xlabel("Year",fontsize=22, **hfont)
ax1.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 12)
ax2.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 12)
# y-axis
ax1.set_ylabel("Income $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax1.set_ylim(-.5E9,4E9)
ax2.set_ylabel("SST anomaly $^\circ C$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='lightgrey')
ax2.set_ylim(-2,15)
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
c, = ax2.plot(yo, color="lightgrey")
# x-axis
plt.xlim(2,tmax)
plt.xlabel("Year",fontsize=22, **hfont)
ax1.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 12)
ax2.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 12)
# y-axis
ax1.set_ylabel("Income $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax1.set_ylim(-0.5E9,2.5E9)
ax2.set_ylabel("Proportion of migrated squid", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='lightgrey')
ax2.set_ylim(-0.2,1)
# adjusting labels and plot size
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(handles=[a,b], loc='best', fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R4_support2.png',dpi=500)
plt.show()
