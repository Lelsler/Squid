#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy import stats
from pandas import *
from scipy import signal

#### Model w/o relationships ###################################################
flag = 1 # 0 = BEM; 1 = MLM, # 2 = BLM
migrate = 0 # 0 = use discrete function, 1 = use continuous function, 2 = use data

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
a4 = 1 # amplitude control
# catchability
k = -0.0318 # catchability slope
l = 0.0018 # catchability intersect
qc = 0.1 # catchability constant
# migration
lamda = 100
alpha = 3.9798459165637236e-15
# trader cooperation
delta = 1 # slope of trader cooperation

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
L = np.zeros(tmax) # pay gap between fishers and traders

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

### Load dataset ###############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_all_plot_data.xlsx', sheetname='Sheet1')
yr = df1['year'] # year
VolAll = df1['C_ALL'] # catch data all locations datamares
PrAll = df1['pf_ALL'] # price data all locations datamares
ml = df1['ML'] # mantle length average per year
ys = df1['M_new'] # migration data from long catch timeseries
ssh = df1['sst_anom'] # SST anomaly

### catchability scaling
def translate(a0, a1, a2, a3, a4, qc):
    for t in np.arange(0,tmax-10): # this assumes that by 2025 temperatures are high so that q = 0
        L[t]= a0 +a1 *(t+1990) +a4* (a2 *np.cos(t+1990) + a3 *np.sin(t+1990))
    return L

for t in np.arange(0,tmax-10): # this assumes that by 2015 temperatures are high so that q = 0
     T = translate(a0, a1, a2, a3, a4, qc)

Tmin = min(T)
Tmax = max(T)

q = qc* ((Tmax-T)/(Tmax-Tmin))

### continuous migration
# this set-up assumes that the proportion of migrated squid is triggered by an environmental signal, the proportion is contingent on the strength (i.e. amplitude) of the signal
xx = np.linspace(1991,2025,1000) # 100 linearly spaced time steps in 35 years
# zz = np.zeros(1000) # array to fill in migration calculations
ko = np.exp(lamda*(a2*np.cos(xx)-a3*np.sin(xx))) # calculate the alpha parameter
alpha = 1/max(ko) # calculate the alpha parameter
# for i in np.arange(0,1000):
#     zz[i] = alpha * np.exp(lamda*a4*(a2*np.cos(xx[i])-a3*np.sin(xx[i]))) # run migration timeseries
#
# peaks, _ = signal.find_peaks(zz) # extract peaks from migration timeseries
#
# xx = np.around(xx, decimals=0) # remove decimals from years
# x= xx[peaks] # keep only peak years
# z= zz[peaks] # keep only peak values

################################################################################
###########################  MODEL FILE  #######################################
################################################################################

### Define Model ###############################################################
def model(a0, a1, a2, a3, a4, Tmin, Tmax, qc, delta, g, K, h1, h2, gamma, beta, kappa, sigma, alpha, lamda):
    # continuous migration
    xx = np.linspace(1991,2025,1000) # 100 linearly spaced time steps in 35 years
    zz = np.zeros(1000) # array to fill in migration calculations
    # ko = np.exp(lamda*a4*(a2*np.cos(xx)-a3*np.sin(xx))) # calculate the alpha parameter
    # alpha = 1/max(ko) # calculate the alpha parameter
    for i in np.arange(0,1000):
        zz[i] = alpha * np.exp(lamda*a4*(a2*np.cos(xx[i])-a3*np.sin(xx[i]))) # run migration timeseries

    peaks, _ = signal.find_peaks(zz) # extract peaks from migration timeseries

    xx = np.around(xx, decimals=0) # remove decimals from years
    x= xx[peaks] # keep only peak years
    z= zz[peaks] # keep only peak values

    for t in np.arange(1,tmax):
        time = t + 1990
        ##### sst anomaly trend
        T[t]= a0 +a1 *(t+1990) +a4* (a2 *np.cos(t+1990) + a3 *np.sin(t+1990))

        # catchability (mantle length)
        q[t] = qc* ((Tmax-T[t])/(Tmax-Tmin))

        if q[t] > qc: # check catchability is in bound - we do not report because we expect high and low values by changing a1 and a4
            q[t] = qc
        elif q[t] < 0:
            q[t] = 0

        # migration of squid
        if any(time == x): # if the current year and a year of migration correspond
            lo = np.where(time == x)
            M[t] = z[lo] # use the migration value of a given year
        else:
            M[t] = 0 # else set migration to a minimum

        if M[t] > 1:
            M[t] = 1

        ##### trader cooperation
        R[t]= np.exp(-delta* M[t])

        ##### squid population
        if flag == 0: # squid population BEM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - qc *E[t-1] *S[t-1]
        if flag >= 1: # squid population MLM, BLM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        ## switch between models ##
        if flag == 0: # effort BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -sigma *E[t-1]
            E[t] = h1 *Escal[t] + h2
        if flag >= 1: # effort MLM, BLM
            Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -sigma *E[t-1]
            E[t] = h1 *Escal[t] + h2

        if E[t] > 1:
            E[t] = 1
            print "E high"
        elif E[t] < 0:
            E[t] = 0
            print "E low"

        ## switch between models ##
        if flag == 0: # catch BEM
            C[t] = qc *E[t] *S[t]
        if flag >= 1: # catch MLM, BLM
            C[t] = q[t] *E[t] *S[t]

        if C[t] <= 0: # avoid infinity calculations
            C[t]= 1

        #### market price
        p_m[t] = gamma* (C[t])**(-beta)

        if p_m[t]>= 99366:
            p_m[t]= 99366
            print "pm high"

        #### price for fishers
        if flag == 0: # BEM
            # price for fishers
            p_f[t] = p_m[t] -kappa
        if flag == 2: # BLM
            p_f[t] = p_m[t] -kappa
        if flag == 1: #MLM
            p_f[t] = (p_m[t] -kappa) *(1-R[t]) +R[t] *((sigma *E[t])/C[t])

        if p_f[t] >= (p_m[t] -kappa): # limit fishers price
            p_f[t] = (p_m[t] -kappa)

        ##### revenue
        # revenue of fishers
        I_f[t] = C[t] *p_f[t] -sigma *E[t]
        # revenue of traders
        I_t[t] = C[t] *p_m[t] -I_f[t] -kappa
        # income gap
        G[t] = I_f[t]/I_t[t]

    return T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G


################################################################################
#############################  RUN MODEL FILE  #################################
################################################################################

##### Run the model ############################################################
a1 = np.arange(0.0195,0.021,0.0000075) # trend. steps to test a1 parameter
a4 = np.arange(0.9,1.1,0.001) # amplitude. steps to test a4 parameter

##### Initiate arrays ##########################################################
cat = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save catches in each time period of each simulation
pri = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save prices for fishers in each time period of each simulation
mar = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save market prices in each time period of each simulation
gap1 = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save price gap mean in each time period of each simulation
gap2 = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save price gap stdv in each time period of each simulation
gap3 = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save icnome gap mean in each time period of each simulation
tem = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save T in each time period of each simulation
mig = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save migrate squid in each time period of each simulation
cco = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save catchability in each time period of each simulation
pop = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save squid population in each time period of each simulation
eff = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save effort in each time period of each simulation
rff = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save revenue fishers in each time period of each simulation
rtt = np.zeros((a1.shape[0],a4.shape[0])) # matrix to save revenue traders in each time period of each simulation

for i in np.arange(0,a1.shape[0]):
    for j in np.arange(0,a4.shape[0]):
        T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G = model(a0, a1[i], a2, a3, a4[j], Tmin, Tmax, qc, delta, g, K, h1, h2, gamma, beta, kappa, sigma, alpha, lamda)

        gap1[i,j]= np.mean(p_f/p_m)
        gap2[i,j]= np.std(p_f/p_m)
        gap3[i,j]= np.mean(I_f/I_t)

        cat[i,j]= np.mean(C)
        pri[i,j]= np.mean(p_f)
        mar[i,j]= np.mean(p_m)

        tem[i,j]= np.mean(T)
        mig[i,j]= np.mean(M)
        cco[i,j]= np.mean(q)
        pop[i,j]= np.mean(S)
        eff[i,j]= np.mean(E)

        rff[i,j]= np.mean(I_f)
        rtt[i,j]= np.mean(I_t)

# np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/parameter_sweep_gap.npy", gap1)
# np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/parameter_sweep_pf.npy", pri)
# np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/parameter_sweep_C.npy", cat)

################################################################################
################################  PLOT FILE  ###################################
################################################################################

# gap1 = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/parameter_sweep_gap.npy")
# pri = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/parameter_sweep_pf.npy")
# cat = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/parameter_sweep_C.npy")

hfont = {'fontname':'Helvetica'}

##### Plot 1 ###################################################################
## define dimensions
y = a1 #  y axis
x = a4 #  x axis
z = gap1 #  output data
## sub plot
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
ax = fig1.add_subplot(gs[0,0])
pcObject = ax.pcolormesh(x,y,z)
plt.pcolormesh(x, y, z, cmap="Spectral")
## set y-axis
ax.set_ylabel('Trend SST anomalies $^\circ C$', fontsize = 22, **hfont)
plt.yticks(np.arange(0.0195,0.021,step=0.0005), ('0.0195', '0.02', '0.0205', '0.021'),fontsize=14)
plt.ylim(0.0195,0.021)
## set x-axis
ax.set_xlabel('Amplitude  SST anomalies $^\circ C$', fontsize = 22, **hfont)
plt.xticks(np.arange(0.9,1.2,step=0.05), ('0.9', '0.95', '1.0', '1.05', '1.1'),fontsize=14)
plt.xlim(0.9,1.1)
## colorbar
cb = plt.colorbar()
cb.set_label('Mean price gap', rotation=270, labelpad=40, fontsize = 22, **hfont)
cb.ax.tick_params(labelsize=12)
## save and show
# fig1.savefig("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/parameter_gap.png",dpi=300)
plt.show()


## define dimensions
y = a1 #  y axis
x = a4 #  x axis
z = rff #  output data
## sub plot
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
ax = fig1.add_subplot(gs[0,0])
pcObject = ax.pcolormesh(x,y,z)
plt.pcolormesh(x, y, z, cmap="Spectral")
## set y-axis
ax.set_ylabel('Trend of SST anomalies $^\circ C$', fontsize = 22, **hfont)
# plt.tick_params(axis='y', which='major', labelsize=14)
plt.yticks(np.arange(0.0195,0.021,step=0.0005), ('0.0195', '0.02', '0.0205', '0.021'),fontsize=14)
plt.ylim(0.0195,0.021)
## set x-axis
ax.set_xlabel('Amplitude of SST anomalies $^\circ C$', fontsize = 22, **hfont)
plt.xticks(np.arange(0.9,1.2,step=0.05), ('0.9', '0.95', '1.0', '1.05', '1.1'),fontsize=14)
plt.xlim(0.9,1.1)
## colorbar
cb = plt.colorbar()
cb.set_label('Fishers income $MXN$', rotation=270, labelpad=40, fontsize = 22, **hfont)
cb.ax.tick_params(labelsize=12)
## save and show
# fig1.savefig("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/parameter_RF.png",dpi=300)
plt.show()


################################################################################
###################################  SI PLOTS ##################################
################################################################################

## define dimensions
y = a1 #  y axis
x = a4 #  x axis
z = cat #  output data
## sub plot
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
ax = fig1.add_subplot(gs[0,0])
pcObject = ax.pcolormesh(x,y,z)
plt.pcolormesh(x, y, z, cmap="Spectral")
## set y-axis
ax.set_ylabel('Trend SST anomalies $^\circ C$', fontsize = 22, **hfont)
plt.yticks(np.arange(0.0195,0.021,step=0.0005), ('0.0195', '0.02', '0.0205', '0.021'),fontsize=14)
plt.ylim(0.0195,0.021)
## set x-axis
ax.set_xlabel('Amplitude SST anomalies $^\circ C$', fontsize = 22, **hfont)
plt.xticks(np.arange(0.9,1.2,step=0.05), ('0.9', '0.95', '1.0', '1.05', '1.1'),fontsize=14)
plt.xlim(0.9,1.1)
## colorbar
cb = plt.colorbar()
cb.set_label('Catches $tons$', rotation=270, labelpad=40, fontsize = 22, **hfont)
cb.ax.tick_params(labelsize=12)
## save and show
# fig1.savefig("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/parameter_C.png",dpi=300)
plt.show()


################################################################################
#############################  RUN TIMESERIES ##################################
################################################################################
#
# OUT1 = np.zeros(T.shape[0])
# OUT2 = np.zeros(T.shape[0])
# OUT3 = np.zeros(T.shape[0])
# OUT4 = np.zeros(T.shape[0])
# OUT5 = np.zeros(T.shape[0])
# OUT6 = np.zeros(T.shape[0])
# OUT7 = np.zeros(T.shape[0])
# OUT8 = np.zeros(T.shape[0])
# OUT9 = np.zeros(T.shape[0])
#
# for i in np.arange(0,tmax):
#     T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G = model(a0, a1, a2, a3, a4, k, l, qc, delta, g, K, h1, h2, gamma, beta, kappa, sigma, alpha, lamda)
#     OUT1[i]= p_f[i]
#     OUT2[i]= C[i]
#     OUT3[i]= T[i]
#     OUT4[i]= M[i]
#     OUT5[i]= q[i]
#     OUT6[i]= S[i]
#     OUT7[i]= E[i]
#     OUT8[i]= p_m[i]
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
# ax1.plot(T, label = "T", color="orange")
# ax1.set_title('Temperature')
# ax1.legend(loc="best")
# # axis 2
# ax2.plot(M, label = "migration", color="orange")
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
# ax4.plot(p_m, label = "market price", color="steelblue")
# ax4.legend(loc="best")
# # Fine-tune figure; make subplots close to each other and hide x ticks for
# # all but bottom plot.
# f.subplots_adjust(hspace=0.2)
# plt.show()
