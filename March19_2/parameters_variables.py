#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # needed for parameter_sweep
import seaborn as sns
import pandas as pd
from scipy import stats
from pandas import *

#################### CHOICES ###################################################
### NEW
timeseries = 1 # 0 = not timeseries simulation; 1 = timeseries simulation
flag = 1 # 0 = BEM; 1 = MLM, # 2 = BLM
mantle = 1 # 0 = use mantle length data, 1 = use SST anomaly

# intervention
inter = 1 # 0 = not intervention simulation; 1 = intervention simulation
competition = 1 # 0 = no intervention; 1 = intervention competition
demand = 0 # 0 = no intervention; 1 = intervention demand
# plots
intervention = 0 # 0 = competition intervention; 1 = demand BEM; 2 = demand BLM; 3= demand MLM

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

# catchability
k = -0.0318 # catchability slope
l = 0.0018 # catchability intersect
qc = 0.1 # catchability constant

# migration
alpha = 1/251266009027660.94 # equivalent to max(np.exp(lamda*(a2*np.cos(xo[i])-a3*np.sin(xo[i]))))
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
T[0] = -0.80 # for SST anomaly
q[0] = 0.05 # squid catchability
M[0] = 0.5 # proportion of migrated squid
R[0] = 0.5 # trader cooperation
S[0] = 610075 # size of the squid population, mean
E[0] = 0.5 # fishing effort
C[0] = 60438 # squid catch mean
p_m[0] = 52035 # mean p_m comtrade, rounded
p_f[0] = 8997 # mean p_f datamares, rounded

####### Pre calculations ########################################################
### catchability scaling
def translate(a0, a1, a2, a3, qc):
    T[t]= a0 +a1 *(t+1990) +a2 *np.cos(t+1990) + a3 *np.sin(t+1990)
    return T

for t in np.arange(0,tmax-10): # this assumes that by 2015 temperatures are high so that q = 0
     T = translate(a0, a1, a2, a3, qc)

Tmin = min(T)
Tmax = max(T)

q = qc-(qc* ((T-Tmin)/(Tmax-Tmin)))

### continuous migration
xo = np.linspace(1991,2025,1000) # 100 linearly spaced numbers, time
ye = np.zeros(1000) # array to fill in migration calculations
xe = np.zeros(1000)
ko = np.exp(lamda*(a2*np.cos(xo)-a3*np.sin(xo)))
alpha = 1/max(ko)
for i in np.arange(0,1000):
    ye[i] = alpha* np.exp(lamda*(a2*np.cos(xo[i])-a3*np.sin(xo[i])))
    if ye[i] > 0.9:
         xe[i] = xo[i]

Mmax = max(ye)
Mmin = min(ye)

xe = np.around(xe, decimals=0)
plt.plot(xo,ye) # migration timeseries
plt.show()

#### Load dataset  #############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_all_plot_data.xlsx', sheetname='Sheet1')
y = df1['year'] # year
pfg = df1['pf_GY'] # fishers price Guaymas
pfs = df1['pf_SR'] # fishers price Santa Rosalia
pfr = df1['pf_RM'] # fishers price other offices
cg = df1['pf_GY'] # catch Guaymas
cs = df1['pf_SR'] # catch Santa Rosalia
cr = df1['pf_RM'] # catch other offices
C = df1['C'] # catches total
sst = df1['sst_anom'] # SST anomaly
ML = df1['ML'] # mantle length - used in timeseries
ys = df1['M'] # migration - used in interventions

### timeseries
if timeseries ==1:
    tmax = 15 # New max time

################################################################################
###########################  MODEL FILE  #######################################
################################################################################

def model(a0, a1, a2, a3, k, l, qc, Tmin, Tmax, Mmax, Mmin, delta, alpha, g, K, h1, h2, gamma, beta, kappa, sigma):
    for t in np.arange(1,tmax):

        # SST anomaly
        if timeseries == 1:
            time = t + 2001
            T[t]= a0 +a1 *(time) +a2 *np.cos(time) +a3 *np.sin(time) # timeseries
        else:
            time = t + 1990
            T[t]= a0 +a1 *(time) +a2 *np.cos(time) +a3 *np.sin(time) # interventions, parameter_sweep

        # catchability (mantle length)
        q[t] = qc - (qc* ((T[t]-Tmin)/(Tmax-Tmin)))

        if mantle == 0: # timeseries # change calculation to data once available
            if ml[t] == 1:
                q[t] = qc - (qc* ((T[t]-Tmin)/(Tmax-Tmin)))
            else:
                ML[t]= ml[t]
                q[t]= l+ ML[t]*k

        if q[t] > qc: # check catchability is in bound and report
            q[t] = qc
            print "q high"
        elif q[t] < 0:
            q[t] = 0
            print "q low"

        # migration of squid
        if any(time == xe):
            M[t] = Mmax
        else:
            M[t] = Mmin # run with continuous function

        # trader cooperation
        R[t]= np.exp(-delta* M[t])

        if inter == 1:
            if t <= timeInt:
                R[t] = np.exp(-delta* M[t])
            else:
                if competition == 0:
                    R[t] = np.exp(-delta* M[t])
                if competition == 1:
                    F[t] = F[t-1]- (i_e * R[t-1])
                    R[t] = F[t]+ np.exp(-delta* M[t])

        # squid population
        if flag == 0: # BEM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - qc *E[t-1] *S[t-1]
        if flag >= 1: # MLM, BLM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        # effort
        if flag == 0: #  BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -sigma *E[t-1]
        if flag >= 1: # effort MLM, BLM
            Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -sigma *E[t-1]

        E[t] = h1 *Escal[t] + h2 # linear scaling of effort

        if E[t] > 1: # check effort is in bound and report
            E[t] = 1
            print "E high"
        elif E[t] < 0:
            E[t] = 0
            print "E low"

        # catch
        if flag == 0: # BEM
            C[t] = qc *E[t] *S[t]
        if flag >= 1: # MLM, BLM
            C[t] = q[t] *E[t] *S[t]

        if C[t] <= 0: # check catch is positive or assign positive
            C[t]= 1

        # market price
        p_m[t] = gamma* (C[t])**(-beta)
        if inter == 1:
            if t <= timeInt:
                p_m[t] = gamma* (C[t])**(-beta)
            else:
                if demand == 0:
                    p_m[t] = gamma* (C[t])**(-beta)
                if demand == 1:
                    p_m[t] = (gamma *(1+ i_e*0.4 *(t -timeInt))) *(C[t])**(-beta)

        if p_m[t]>= 99366: # check market price is in bound and report
            p_m[t]= 99366
            print "pe high"


        # fishers price
        if flag == 0: # BEM
            p_f[t] = p_m[t] -kappa
        if flag == 2: # BLM
            p_f[t] = p_m[t] -kappa
        if flag == 1: # MLM
            p_f[t] = (p_m[t] -kappa) *(1-R[t]) +R[t] *(sigma *E[t])/C[t] # where (sigma *E[t])/C[t] constitutes minimum fishers price

        if p_f[t] >= (p_m[t] -kappa): # check fishers price is in bound and report (i.e. maximum price traders would pay)
            p_f[t] = (p_m[t] -kappa)
            print "high fishers price"

        # income
        I_f[t] = C[t] *p_f[t] -sigma *E[t] # fishers
        I_t[t] = C[t] *p_m[t] -I_f[t] -kappa # traders
        G[t] = I_f[t]/I_t[t] # income gap

    return T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G

################################################################################
###########################  RUN MODEL FILE  ###################################
################################################################################
#
# ##### interventions ###########################################################
# OUT1 = np.zeros(T.shape[0])
# OUT2 = np.zeros(T.shape[0])
# OUT3 = np.zeros(T.shape[0])
# OUT4 = np.zeros(T.shape[0])
# OUT5 = np.zeros(T.shape[0])
# OUT6 = np.zeros(T.shape[0])
# OUT7 = np.zeros(T.shape[0])
# OUT8 = np.zeros(T.shape[0])
# OUT9 = np.zeros(T.shape[0])
# OUT10 = np.zeros(T.shape[0])
# OUT11 = np.zeros(T.shape[0])
# OUT12 = np.zeros(T.shape[0])
#
# ##### Run the model ############################################################
# for i in np.arange(0,tmax):
#         T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G = model(a0, a1, a2, a3, k, l, qc, Tmin, Tmax, Mmax, Mmin, delta, alpha, g, K, h1, h2, gamma, beta, kappa, sigma)
#         OUT3[i]= T[i]
#         OUT5[i]= q[i]
#         OUT4[i]= M[i]
#         OUT6[i]= S[i]
#         OUT7[i]= E[i]
#         OUT1[i]= p_f[i]
#         OUT8[i]= p_m[i]
#         OUT2[i]= C[i]
#         OUT9[i]= G[i]
#         OUT10[i]= I_f[i]
#         OUT11[i]= I_t[i]

#
# ##### Initiate arrays ##########################################################
# sim = np.arange(0,10) # number of simulations
# x = np.zeros(7) # set array to save parameters
# par = np.zeros((sim.shape[0],x.shape[0])) # matrix to save parameter values of each simulation
# cat = np.zeros((sim.shape[0],T.shape[0])) # matrix to save catches in each time period of each simulation
# pri = np.zeros((sim.shape[0],T.shape[0])) # matrix to save prices in each time period of each simulation
#
# ### extra variables to monitor
# tem = np.zeros((sim.shape[0],T.shape[0])) # matrix to save T in each time period of each simulation
# mig = np.zeros((sim.shape[0],T.shape[0])) # matrix to save migrate squid in each time period of each simulation
# cco = np.zeros((sim.shape[0],T.shape[0])) # matrix to save catchability in each time period of each simulation
# pop = np.zeros((sim.shape[0],T.shape[0])) # matrix to save squid population in each time period of each simulation
# eff = np.zeros((sim.shape[0],T.shape[0])) # matrix to save effort in each time period of each simulation
# mar = np.zeros((sim.shape[0],T.shape[0])) # matrix to save market prices in each time period of each simulation
#
# gap = np.zeros((sim.shape[0],T.shape[0])) # matrix to save revenue gap
# rvf = np.zeros((sim.shape[0],T.shape[0])) # matrix to save income fishers
# rvt = np.zeros((sim.shape[0],T.shape[0])) # matrix to save income traders
# rva = np.zeros((sim.shape[0],T.shape[0])) # matrix to save revenue fishery
#
#
# ##### Timeseries ############################################################
# for j in range(0,sim.shape[0]): # draw randomly a float in the range of values for each parameter
#     qc = np.random.uniform(0.01, 0.5)
#     delta = np.random.uniform(0.5, 1.5)
#     g = np.random.uniform(0, 2.9)
#     gamma = np.random.uniform(20000, 51000)
#     beta = np.random.uniform(0.01, 0.1)
#     kappa = np.random.uniform(1000, 2148)
#     sigma = np.random.uniform(50907027, 212300758)
#
#     x = [qc, delta, g, gamma, beta, kappa, sigma]
#     par[j] = x
#
#     OUT1 = np.zeros(T.shape[0])
#     OUT2 = np.zeros(T.shape[0])
#     OUT3 = np.zeros(T.shape[0])
#     OUT4 = np.zeros(T.shape[0])
#     OUT5 = np.zeros(T.shape[0])
#     OUT6 = np.zeros(T.shape[0])
#     OUT7 = np.zeros(T.shape[0])
#     OUT8 = np.zeros(T.shape[0])
#     OUT9 = np.zeros(T.shape[0])
#     OUT10 = np.zeros(T.shape[0])
#     OUT11 = np.zeros(T.shape[0])
#     OUT12 = np.zeros(T.shape[0])
#
#     for i in np.arange(1,tmax):
#             T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G = model(a0, a1, a2, a3, k, l, qc, Tmin, Tmax, Mmax, Mmin, delta, alpha, g, K, h1, h2, gamma, beta, kappa, sigma)
#             OUT1[i]= p_f[i]
#             OUT2[i]= C[i]
#             OUT3[i]= T[i]
#             OUT4[i]= M[i]
#             OUT5[i]= q[i]
#             OUT6[i]= S[i]
#             OUT7[i]= E[i]
#             OUT8[i]= p_m[i]
#             OUT9[i]= G[i]
#             OUT10[i]= I_f[i]
#             OUT11[i]= I_t[i]
#             pri[j,i] = OUT1[i]
#             cat[j,i] = OUT2[i]
#             tem[j,i] = OUT3[i]
#             mig[j,i] = OUT4[i]
#             cco[j,i] = OUT5[i]
#             pop[j,i] = OUT6[i]
#             eff[j,i] = OUT7[i]
#             mar[j,i] = OUT8[i]
#             gap[j,i] = OUT9[i]
#             rvf[j,i] = OUT10[i]
#             rvt[j,i] = OUT11[i]
#             rva[j,i] = OUT12[i]
#
# # initiate variables for 95% confidence interval
# lowC = np.zeros(y.shape[0])
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
#
