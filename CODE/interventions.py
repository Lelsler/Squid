#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy
from pandas import *
from scipy import signal

#################### CHOICES ###################################################
flag = 1 # 0 = BEM; 1 = MLM, # 2 = EDM
# SST or SSTres: parameters, initial condition for T, and MC parameter ranges
intervention = 3 # 0 = competition intervention; 1 = demand BEM; 2 = demand EDM; 3= demand MLM
competition = 0 # 0 = no intervention; 1 = intervention competition
demand = 1 # 0 = no intervention; 1 = intervention demand

# for competition int: 1010
# for demand BEM: 0101
# for demand EDM: 2201
# for demand MLM: 1301

### Parameters #################################################################
tmax = 35 # model run, years
# population
K = 1208770 # carrying capacity
g = 1.4 # population growth rate
# fishing effort
sigma = 107291548 # cost of fishing
h1 = 2E-10 # scale E from Escal €[-3,10E+09; 1,60E+09]
h2 = 0.6596 # scale E from Escal €[-3,10E+09; 1,60E+09]
# prices
gamma = 49200 # maximum demand
beta = 0.0736 # slope of demand-price function
kappa = 1776.25 # cost of processing
# temperature
a0 = -40.9079 # intersection y
a1 = 0.020464 # trend
a2 = 0.165387 # periodicity
a3 = -0.287384 # periodicity
a4 = 1 # amplitude - unlike for the other models we need an a4 parameter here to change the amplitude across sin and cos
# catchability
k = -0.0318 # catchability slope
l = 0.0018 # catchability intersect
qc = 0.1 # catchability constant
# migration
lamda = 100
# trader cooperation
delta = 1 # slope of trader cooperation

### interventions ###
i_e = 0.1 # increment of decrease in traders cooperation per unit investment
timeInt = 15 # year to start intervention

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
p_f = np.zeros(tmax) # price for fishers
I_f = np.zeros(tmax) # revenue of fishers
I_t = np.zeros(tmax) # revenue of traders
G = np.zeros(tmax) # pay gap between fishers and traders

### interventions ###
F = np.zeros(tmax) # intercept of traders cooperation

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

#### Load dataset  #############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_all_plot_data.xlsx', sheetname='Sheet1')
y = df1['year'] # year
pf = df1['pf_ALL'] # price data all locations datamares
ct = df1['C_ALL'] # catch data all locations datamares
ssh = df1['sst_anom'] # SST anomaly
ml = df1['ML'] # mantle length average per year
ys = df1['M_new'] # migration data from long catch timeseries

####### Pre calculations ########################################################
### catchability scaling
def translate(a0, a1, a2, a3, a4, qc):
    T[t]= a0 +a1 *(t+1990) +a4* (a2 *np.cos(t+1990) + a3 *np.sin(t+1990)) # temperature function
    return T

for t in np.arange(0,tmax-10): # this assumes that by 2015 temperatures are high so that q = 0
     T = translate(a0, a1, a2, a3, a4, qc)

Tmin = min(T)
Tmax = max(T)

q = qc* ((Tmax-T)/(Tmax-Tmin))

### continuous migration
# this set-up assumes that the proportion of migrated squid is triggered by an environmental signal, the proportion is contingent on the strength (i.e. amplitude) of the signal
xx = np.linspace(1991,2025,1000) # 100 linearly spaced time steps in 35 years
zz = np.zeros(1000) # array to fill in migration calculations
ko = np.exp(lamda*(a2*np.cos(xx)-a3*np.sin(xx))) # calculate the alpha parameter
alpha = 1/max(ko) # calculate the alpha parameter
for i in np.arange(0,1000):
    zz[i] = alpha * np.exp(lamda*a4*(a2*np.cos(xx[i])-a3*np.sin(xx[i]))) # run migration timeseries

plt.plot(zz)
plt.show()

peaks, _ = signal.find_peaks(zz) # extract peaks from migration timeseries

xx = np.around(xx, decimals=0) # remove decimals from years
x= xx[peaks] # keep only peak years
z= zz[peaks] # keep only peak values
print z, x


################################################################################
###########################  MODEL FILE  #######################################
################################################################################

### Define Model ###############################################################
def model(a0, a1, a2, a3, a4, k, l, qc, Tmin, Tmax, delta, alpha, i_e, timeInt, g, K, h1, h2, gamma, beta, kappa, sigma):
    for t in np.arange(1,tmax):
        T[t]= a0 +a1 *(t+1990) +a4* (a2 *np.cos(t+1990) + a3 *np.sin(t+1990))
        time = 1990 +t

        # catchability
        q[t] = qc* ((Tmax-T[t])/(Tmax-Tmin))
        print q[t]

        if q[t] > qc: # check catchability is in bound and report
            q[t] = qc
            print "q high" # we do not expect any high q
        elif q[t] < 0:
            q[t] = 0
            print "q low" # we expect low q: Tmax is calculated for the year 2015

        # migration of squid
        if any(time == x): # if the current year and a year of migration correspond
            lo = np.where(time == x)
            M[t] = z[lo] # use the migration value of a given year
        else:
            M[t] = 0 # else set migration to a minimum

        if M[t] > 1:
            M[t] = 1
            print "M high"

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
        if flag >= 1: # squid population MLM, EDM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        #### switch between models ####
        if flag == 0: # effort BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -sigma *E[t-1]
            E[t] = h1 *Escal[t] + h2

        if flag >= 1: # effort MLM, EDM
            Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -sigma *E[t-1]
            # fishing effort scaled
            E[t] = h1 *Escal[t] + h2

        if E[t] > 1:
            E[t] = 1
            print "E high"
        elif E[t] < 0:
            E[t] = 0
            print "E low"

        #### switch between models ####
        if flag == 0: # catch BEM
            C[t] = qc *E[t] *S[t]
        if flag >= 1: # catch MLM, EDM
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
            print "pm high"

        #### switch between models ####
        if flag == 0: # BEM
            p_f[t] = p_m[t] -kappa # fishers price
        if flag == 2: # EDM
            p_f[t] = p_m[t] -kappa  # fishers price
        if flag == 1: #MLM
            p_f[t] = (p_m[t] -kappa) *(1-R[t]) +R[t] *((sigma *E[t])/C[t])

        if p_f[t] >= (p_m[t] -kappa): # limit price of fishers
            p_f[t] = (p_m[t] -kappa)

        # revenue of fishers
        I_f[t] = C[t] *p_f[t] -sigma *E[t]
        # revenue of traders
        I_t[t] = C[t] *p_m[t] -I_f[t] -kappa
        # income gap
        G[t] = I_f[t]/I_t[t]

    return T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G

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

##### Run the model ############################################################
for i in np.arange(1,tmax):
        T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G = model(a0, a1, a2, a3, a4, k, l, qc, Tmin, Tmax, delta, alpha, i_e, timeInt, g, K, h1, h2, gamma, beta, kappa, sigma)
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
# x-axis
#plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(1,tmax)
plt.xlabel("time $years$",fontsize=22, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
# plt.ylim(0,4E9)
plt.ylabel("Income and revenue $MXN$" ,fontsize=22, **hfont)
plt.legend(handles=[a,b], loc='best', fontsize=14)
# load and show
plt.show()

### advanced plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
a, = ax1.plot(OUT10, label = "Fishers income", color = 'steelblue')
b, = ax1.plot(OUT11, label = "Traders income", color = 'orange')
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
# ax1.set_ylim(0,3E9)
ax2.set_ylabel("SST anomaly $C$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='lightgrey')
# ax2.set_ylim(-6,0)
# adjusting labels and plot size
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(handles=[a,b,d], loc='best', fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/intervention_support2.png',dpi=300)
plt.show()


##### Save data  ###############################################################
# if intervention == 0: # intervention competition
#     print "intervention competition"
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_MLM_pf.npy", OUT1)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_MLM_pe.npy", OUT8)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_MLM_RF.npy", OUT10)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_MLM_RT.npy", OUT11)
# if intervention == 1: # intervention demand BEM
#     print "intervention demand BEM"
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_BEM_pf.npy", OUT1)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_BEM_pe.npy", OUT8)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_BEM_RF.npy", OUT10)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_BEM_RT.npy", OUT11)
# if intervention == 2: # intervention demand EDM
#     print "intervention demand EDM"
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_EDM_pf.npy", OUT1)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_EDM_pe.npy", OUT8)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_EDM_RF.npy", OUT10)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_EDM_RT.npy", OUT11)
# if intervention == 3: # intervention demand MLM
#     print "intervention demand MLM"
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_MLM_pf.npy", OUT1)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_MLM_pe.npy", OUT8)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_MLM_RF.npy", OUT10)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_MLM_RT.npy", OUT11)

################################################################################
###########################  PLOT FILE  ########################################
################################################################################

# ##### Load data  ###############################################################
# competition intervention
cPE = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_MLM_pe.npy")
cPF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_MLM_pf.npy")
cRF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_MLM_RF.npy")
cRT = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_MLM_RT.npy")
# intervention demand BEM
dBPE = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_BEM_pe.npy")
dBPF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_BEM_pf.npy")
dBRF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_BEM_RF.npy")
dBRT = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_BEM_RT.npy")
# intervention demand EDM
dLPE = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_EDM_pe.npy")
dLPF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_EDM_pf.npy")
dLRF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_EDM_RF.npy")
dLRT = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_EDM_RT.npy")
# intervention demand MLM
dMPE = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_MLM_pe.npy")
dMPF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_MLM_pf.npy")
dMRF = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_MLM_RF.npy")
dMRT = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/intervention_gamma_MLM_RT.npy")


### font ######################################################################
hfont = {'fontname':'Helvetica'}

yr = np.arange(1990,2025,1)

# begin plotting demand intervention
fig = plt.figure()
ax1 = fig.add_subplot(111)
a, = ax1.plot(dMRF, label = "MLM", color = 'steelblue', linestyle='-', linewidth=3)
b, = ax1.plot(dMRT, label = "MLM", color = 'orange', linestyle='-', linewidth=3)
c, = ax1.plot(dLRF, label = "EDM", color = 'steelblue', linestyle='--')
d, = ax1.plot(dLRT, label = "EDM", color = 'orange', linestyle='--')
e, = ax1.plot(dBRF, label = "BEM", color = 'steelblue', linestyle=':')
f, = ax1.plot(dBRT, label = "BEM", color = 'orange', linestyle=':')
# x-axis
# add the second axes using subplot with ML
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
g, = ax2.plot(OUT3, color="silver", linewidth=2)
# x-axis
plt.xlim(2,tmax)
# plt.xlabel("Year",fontsize=22, **hfont)
ax1.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 14)
ax2.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 14)
# y-axis
ax1.set_ylabel("Income $MXN$", rotation=90, labelpad=5, fontsize=22, **hfont)
ax1.set_ylim(-.5E9,3.5E9)
ax1.tick_params(axis='y', labelsize=14)
ax2.set_ylabel("SST anomaly $^\circ C$", rotation=270, color='silver', labelpad=22, fontsize=22, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='silver', labelsize=14)
ax2.set_ylim(-1,19)
# adjusting labels and plot size
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(handles=[b,d,f], loc=2, fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/intervention_demand.pdf',dpi=300)
plt.show()


# begin plotting competition intervention
fig = plt.figure()
ax1 = fig.add_subplot(111)
a, = ax1.plot(cRF, label = "Fisher", color = 'steelblue', linewidth=3)
b, = ax1.plot(cRT, label = "Trader", color = 'orange', linewidth=3)
# x-axis
# add the second axes using subplot with ML
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
c, = ax2.plot(OUT4, color="silver", linewidth=2)
# x-axis
plt.xlim(2,tmax)
# plt.xlabel("Year",fontsize=22, **hfont)
ax1.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 14)
ax2.set_xticklabels(np.arange(1990,2035,5), rotation=45, fontsize= 14)
# y-axis
ax1.set_ylabel("Income $MXN$", rotation=90, labelpad=5, fontsize=22, **hfont)
ax1.set_ylim(-.5E9,2.5E9)
ax1.tick_params(axis='y', labelsize=14)
ax2.set_ylabel("% Pacific landed squid", rotation=270, color='silver', labelpad=22, fontsize=22, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='silver', labelsize=14)
ax2.set_ylim(-0.5,2.5)
# adjusting labels and plot size
plt.gcf().subplots_adjust(bottom=0.15)
plt.legend(handles=[a,b], loc='best', fontsize=14)
# load and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/intervention_competition.pdf',dpi=300)
plt.show()
