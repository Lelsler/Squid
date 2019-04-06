#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy
from pandas import *

#################### CHOICES ###################################################
flag = 2 # 0 = BEM; 1 = MLM, # 2 = EDM
mantle = 0 # 0 = use mantle length input, else use calculation of q(T)

### Parameters #################################################################
tmax = 27 # model run, years
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
lamda = 100
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


### Initial values #############################################################
T[0] = -0.9 # for SST anomaly
q[0] = 0.05 # squid catchability
M[0] = 0.5 # proportion of migrated squid
R[0] = 0.5 # trader cooperation
S[0] = 610075 # size of the squid population, mean
E[0] = 0.5 # fishing effort
C[0] = 60438 # squid catch mean
p_m[0] = 52035 # mean p_e comtrade, rounded
p_f[0] = 8997 # mean p_f datamares, rounded

####### Pre calculations ########################################################
### catchability scaling
def translate(a0, a1, a2, a3, qc):
    T[t]= a0 +a1 *(t+1990) +a2 *np.cos(t+1990) + a3 *np.sin(t+1990)
    return T

for t in np.arange(0,tmax): # this assumes that by 2025 temperatures are high so that q = 0
     T = translate(a0, a1, a2, a3, qc)

Tmin = min(T)
Tmax = max(T)

q = qc* ((Tmax-T)/(Tmax-Tmin))

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
plt.plot(xo,ye)
# plt.show()

### Load dataset ###############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_all_plot_data.xlsx', sheetname='Sheet1')
yr = df1['year'] # year
VolAll = df1['C_ALL'] # catch data all locations datamares
PrAll = df1['pf_ALL'] # price data all locations datamares
ml = df1['ML'] # mantle length average per year
ys = df1['M_new'] # migration data from long catch timeseries
ssh = df1['sst_anom'] # SST anomaly

################################################################################
###########################  MODEL FILE  #######################################
################################################################################


### Define Model ###############################################################
def model(a0, a1, a2, a3, k, l, qc, Tmin, Tmax, Mmax, Mmin, alpha, delta, g, K, h1, h2, gamma, beta, sigma, kappa):
    for t in np.arange(1,tmax):
        time = t + 2001
        T[t]= a0 +a1 *(t+2001) +a2 *np.cos(t+2001) + a3 *np.sin(t+2001)

        # mantle length and catchability
        if mantle == 0:
            if ml[t] == 1:
                q[t] = qc* ((Tmax-T[t])/(Tmax-Tmin))
            else:
                ML[t]= ml[t]
                q[t]= 0.0018 *ML[t] - 0.0318
        if mantle == 1:
            q[t] = qc* ((Tmax-T[t])/(Tmax-Tmin))

        if q[t] > 0.1:
            q[t] = 0.1
            print "q high"
        elif q[t] < 0:
            q[t] = 0
            print "q low"

        # migration of squid
        if mantle == 0: # USE DATA INPUT?
            if ys[t] == 1: # run w/o data
                if any(time == xe):
                        M[t] = Mmax
                else:
                    M[t] = Mmin # run with continuous function
            else:
                M[t]= ys[t] # run with data
        if mantle == 1: # use simulation input
            if any(time == xe):
                    M[t] = Mmax
            else:
                M[t] = Mmin # run with continuous function

        if M[t] > 1:
            M[t] = 1
            print "M high"
        elif M[t] < 0:
            M[t] = 0
            print "M low"

        # trader cooperation
        R[t]= np.exp(-delta* M[t])

        #### switch between models ####
        if flag == 0: # squid population BEM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - qc *E[t-1] *S[t-1]
        if flag >= 1: # squid population MLM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        #### switch between models ####
        if flag == 0: # effort BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -sigma *E[t-1]
            E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]

        if flag >= 1: # effort MLM
            Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -sigma *E[t-1] # c_t is per trip so we need to upscale E hr > fisher > trip
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
        p_m[t] = gamma* (C[t])**(-beta)
        if p_m[t] >= 99366:
            p_m[t] = 99366
            print "pm high"

        #### switch between models ####
        if flag == 0: # BEM
            # price for fishers
            p_f[t] = p_m[t] -kappa
        if flag == 2: # EDM
            p_f[t] = p_m[t] -kappa    # fishers price
        if flag == 1: #MLM
            # price for fishers
            p_f[t] = (p_m[t] -kappa) *(1-R[t]) +R[t] *((sigma *E[t])/C[t])

        if p_f[t] >= (p_m[t] -kappa): # limit fishers price
            p_f[t] = (p_m[t] -kappa)

        # revenue of fishers
        I_f[t] = C[t] *p_f[t] -sigma *E[t]
        # revenue of traders
        I_t[t] = C[t] *p_m[t] -I_f[t] -kappa

        # pay gap
        G[t] = I_f[t]/I_t[t]

    return T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G

################################################################################
###########################  RUN MODEL FILE  ###################################
################################################################################

##### Initiate arrays ##########################################################
sim = np.arange(0,100) # number of simulations
x = np.zeros(5) # set array to save parameters
par = np.zeros((sim.shape[0],x.shape[0])) # matrix to save parameter values of each simulation
cat = np.zeros((sim.shape[0],T.shape[0])) # matrix to save catches in each time period of each simulation
pri = np.zeros((sim.shape[0],T.shape[0])) # matrix to save prices in each time period of each simulation

### extra variables to monitor
tem = np.zeros((sim.shape[0],T.shape[0])) # matrix to save T in each time period of each simulation
mig = np.zeros((sim.shape[0],T.shape[0])) # matrix to save migrate squid in each time period of each simulation
cco = np.zeros((sim.shape[0],T.shape[0])) # matrix to save catchability in each time period of each simulation
pop = np.zeros((sim.shape[0],T.shape[0])) # matrix to save squid population in each time period of each simulation
eff = np.zeros((sim.shape[0],T.shape[0])) # matrix to save effort in each time period of each simulation
mar = np.zeros((sim.shape[0],T.shape[0])) # matrix to save market prices in each time period of each simulation
gap = np.zeros((sim.shape[0],T.shape[0])) # matrix to save revenue gap
rvf = np.zeros((sim.shape[0],T.shape[0])) # matrix to save income fishers
rvt = np.zeros((sim.shape[0],T.shape[0])) # matrix to save income traders


##### Run the model ############################################################
for j in range(0,sim.shape[0]): # draw randomly a float in the range of values for each parameter
    qc = np.random.uniform(0.01, 0.5)
    delta = np.random.uniform(0.5, 1.5)
    g = np.random.uniform(0, 2.9)
    kappa = np.random.uniform(1000, 2148)
    sigma = np.random.uniform(50907027, 212300758)

    x = [qc, delta, g, kappa, sigma]
    par[j] = x

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

    for i in np.arange(1,tmax):
            T, ML, q, M, R, S, E, C, p_m, p_f, I_f, I_t, G = model(a0, a1, a2, a3, k, l, qc, Tmin, Tmax, Mmax, Mmin, alpha, delta, g, K, h1, h2, gamma, beta, sigma, kappa)
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
            pri[j,i] = OUT1[i]
            cat[j,i] = OUT2[i]
            tem[j,i] = OUT3[i]
            mig[j,i] = OUT4[i]
            cco[j,i] = OUT5[i]
            pop[j,i] = OUT6[i]
            eff[j,i] = OUT7[i]
            mar[j,i] = OUT8[i]
            gap[j,i] = OUT9[i]
            rvf[j,i] = OUT10[i]
            rvt[j,i] = OUT11[i]

lowC = np.zeros(yr.shape[0]) # initiate variables for 95% confidence interval
highC = np.zeros(yr.shape[0])
meanC = np.zeros(yr.shape[0])
lowP = np.zeros(yr.shape[0])
highP = np.zeros(yr.shape[0])
meanP = np.zeros(yr.shape[0])

for h in range(0,yr.shape[0]): # calculate the 95% confidence interval
    z = cat[:,h]
    lowC[h] = np.nanmean(z) - ((1.96 * np.nanstd(z))/np.sqrt(np.count_nonzero(~np.isnan(z))))
    highC[h] = np.nanmean(z) + ((1.96 * np.nanstd(z))/np.sqrt(np.count_nonzero(~np.isnan(z))))
    meanC[h] = np.nanmean(z)
    zeta = pri[:,h]
    lowP[h] = np.nanmean(zeta) - ((1.96 * np.nanstd(zeta))/np.sqrt(np.count_nonzero(~np.isnan(zeta))))
    highP[h] = np.nanmean(zeta) + ((1.96 * np.nanstd(zeta))/np.sqrt(np.count_nonzero(~np.isnan(zeta))))
    meanP[h] = np.nanmean(zeta)


### Save data  ###############################################################
if flag == 0:
    print "bem save"
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_BEM_lowC.npy", lowC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_BEM_highC.npy", highC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_BEM_meanC.npy", meanC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_BEM_lowP.npy", lowP)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_BEM_highP.npy", highP)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_BEM_meanP.npy", meanP)
if flag == 1:
    print "mlm save"
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_MLM_lowC.npy", lowC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_MLM_highC.npy", highC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_MLM_meanC.npy", meanC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_MLM_lowP.npy", lowP)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_MLM_highP.npy", highP)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_MLM_meanP.npy", meanP)
if flag == 2:
    print "edm save"
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_EDM_lowC.npy", lowC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_EDM_highC.npy", highC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_EDM_meanC.npy", meanC)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_EDM_lowP.npy", lowP)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_EDM_highP.npy", highP)
    np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_EDM_meanP.npy", meanP)

################################################################################
###########################  PLOT FILE  ########################################
################################################################################

#### Load data  #ts_catch_95_BEM_meanC.npy####################################
NoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesNoR_pf.npy")
NoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesNoR_C.npy")
lowBEM_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_BEM_lowP.npy")
highBEM_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_BEM_highP.npy")
meanBEM_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_BEM_meanP.npy")
lowBEM_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_BEM_lowC.npy")
highBEM_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_BEM_highC.npy")
meanBEM_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_BEM_meanC.npy")

lowMLM_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_MLM_lowP.npy")
highMLM_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_MLM_highP.npy")
meanMLM_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_MLM_meanP.npy")
lowMLM_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_MLM_lowC.npy")
highMLM_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_MLM_highC.npy")
meanMLM_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_MLM_meanC.npy")

lowEDM_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_EDM_lowP.npy")
highEDM_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_EDM_highP.npy")
meanEDM_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_price_95_EDM_meanP.npy")
lowEDM_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_EDM_lowC.npy")
highEDM_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_EDM_highC.npy")
meanEDM_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/ts_catch_95_EDM_meanC.npy")


### New settings ###############################################################
tmax = len(yr) # max time
x = np.arange(0,len(yr))
hfont = {'fontname':'Helvetica'} # font

#####! PLOT MODEL  #############################################################
fig = plt.figure()
fig.subplots_adjust(bottom=0.15, left= 0.15)
# add the first axes using subplot populated with predictions
ax1 = fig.add_subplot(111)
line1, = ax1.plot(meanMLM_P, label = "MLM", color="orange")
line2, = ax1.plot(meanEDM_P, label = "EDM", color="sage")
line3, = ax1.plot(meanBEM_P, label = "BEM", color="steelblue")
line4, = ax1.plot(PrAll, label = "data", color = "indianred", linewidth=4)
ax1.fill_between(x, highMLM_pf, lowMLM_pf, where = highMLM_pf >= lowMLM_pf, facecolor='orange', alpha= 0.3, zorder = 0)
ax1.fill_between(x, highBEM_pf, lowBEM_pf, where = highBEM_pf >= lowBEM_pf, facecolor='steelblue', alpha= 0.3, zorder = 0)
ax1.fill_between(x, highEDM_pf, lowEDM_pf, where = highEDM_pf >= lowEDM_pf, facecolor='sage', alpha= 0.3, zorder = 0)
# add the second axes using subplot with ML
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line5, = ax2.plot(ys, color="silver")
# x-axis
ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 14)
ax1.set_xlim(10,tmax-2)
ax1.set_xlabel("Year",fontsize=20, **hfont)
ax2.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 14)
ax2.set_xlim(10,tmax-2)
ax2.set_xlabel("Year",fontsize=20, **hfont)
# y-axis
ax1.set_ylabel("Prices for fishers $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax1.tick_params(axis='y', labelsize=14)
ax2.set_ylabel("Squid landed outside the GOC", rotation=270, color='silver', labelpad=22, fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='silver', labelsize=14)
ax2.set_ylim(0,1)
plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# legend
plt.legend([line1, line2, line3, line4], ["MLM", "EDM", "BEM", "Data"], loc=2, fontsize= 12)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/ts_price.png',dpi=300)
plt.show()

fig = plt.figure()
fig.subplots_adjust(bottom=0.15, left= 0.15)
# add the first axes using subplot populated with predictions
ax1 = fig.add_subplot(111)
line1, = ax1.plot(meanMLM_C, label = "MLM", color="orange")
line2, = ax1.plot(meanEDM_C, label = "EDM", color="sage")
line3, = ax1.plot(meanBEM_C, label = "BEM", color="steelblue")
line4, = ax1.plot(VolAll, label = "data", color = "indianred", linewidth=4)
ax1.fill_between(x, highMLM_C, lowMLM_C, where = highMLM_C >= lowMLM_C, facecolor='orange', alpha= 0.3, zorder = 0)
ax1.fill_between(x, highBEM_C, lowBEM_C, where = highBEM_C >= lowBEM_C, facecolor='steelblue', alpha= 0.3, zorder = 0)
ax1.fill_between(x, highEDM_C, lowEDM_C, where = highEDM_C >= lowEDM_C, facecolor='sage', alpha= 0.3, zorder = 0)
# add the second axes using subplot with ML
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line5, = ax2.plot(ml, color="silver")
# x-axis
ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 14)
ax1.set_xlim(10,tmax-2)
ax1.set_xlabel("Year",fontsize=20, **hfont)
ax2.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 14)
ax2.set_xlim(10,tmax-2)
ax2.set_xlabel("Year",fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
# y-axis
ax1.set_ylabel("Catch $tons$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax1.set_ylim(0,3E5)
ax1.tick_params(axis='y', labelsize=14)
ax2.set_ylabel("Mantle length $cm$", rotation=270, color='silver', labelpad=22, fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis='y', colors='silver', labelsize=14)
ax2.set_ylim(0,140)
plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# legend
plt.legend([line1, line2, line3, line4], ["MLM", "EDM", "BEM", "Data"], loc=1, fontsize= 12)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/ts_catch.png',dpi=300)
plt.show()

### CALCULATE r squared ########################################################
### price for fishers
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(PrAll[10:-1], meanMLM_P[10:-1])
print("r-squared price MLM:", r_value**2)
scipy.stats.pearsonr(PrAll[10:-1], meanMLM_P[10:-1])

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(PrAll[10:-1], meanBEM_P[10:-1])
print("r-squared price BEM:", r_value**2)
scipy.stats.pearsonr(PrAll[10:-1], meanBEM_P[10:-1])

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(PrAll[10:-1], meanEDM_P[10:-1])
print("r-squared price EDM:", r_value**2)
scipy.stats.pearsonr(PrAll[10:-1], meanEDM_P[10:-1])

### catch
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanMLM_C[10:-1])
print("r-squared catch MLM:", r_value**2)
scipy.stats.pearsonr(VolAll[10:-1], meanMLM_C[10:-1])

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanBEM_C[10:-1])
print("r-squared catch BEM:", r_value**2)
scipy.stats.pearsonr(VolAll[10:-1], meanBEM_C[10:-1])

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanEDM_C[10:-1])
print("r-squared catch EDM:", r_value**2)
scipy.stats.pearsonr(VolAll[10:-1], meanEDM_C[10:-1])


################################################################################
###########################  PLOT TEST - direclty plots from model output  #####
################################################################################

#
# x = np.arange(0,len(yr))
#
# hfont = {'fontname':'Helvetica'}
#
# fig = plt.figure()
# # add the first axes using subplot populated with predictions
# ax1 = fig.add_subplot(111)
# line1, = ax1.plot(meanP, label = "MLM", color="orange")
# line2, = ax1.plot(PrAll, label = "data", color = "indianred")
# ax1.fill_between(x, highP, lowP, where = highP >= lowP, facecolor='orange', alpha= 0.3, zorder = 0)
# # add the second axes using subplot with ML
# ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
# line3, = ax2.plot(ml, color="silver")
# # x-axis
# ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax1.set_xlim(10,tmax-2)
# ax1.set_xlabel("Year",fontsize=20, **hfont)
# ax2.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax2.set_xlim(10,tmax-2)
# ax2.set_xlabel("Year",fontsize=20, **hfont)
# # y-axis
# ax1.set_ylabel("Prices for fishers $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
# ax2.set_ylabel("Mantle length $cm$", rotation=270, color='silver', labelpad=22, fontsize=20, **hfont)
# plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
# # legend
# plt.legend([line1, line2, line3], ["Prediction", "Data", "Mantle length"], fontsize= 11)
# # save and show
# # fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support1MC.png',dpi=300)
# plt.show()
#
#
# fig = plt.figure()
# # add the first axes using subplot populated with predictions
# ax1 = fig.add_subplot(111)
# line1, = ax1.plot(meanC, label = "MLM", color="orange")
# line2, = ax1.plot(VolAll, label = "data", color = "indianred")
# ax1.fill_between(x, highC, lowC, where = highC >= lowC, facecolor='orange', alpha= 0.3, zorder = 0)
# # add the second axes using subplot with ML
# ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
# line3, = ax2.plot(ml, color="silver")
# # x-axis
# ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax1.set_xlim(10,tmax-2)
# ax1.set_xlabel("Year",fontsize=20, **hfont)
# ax2.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax2.set_xlim(10,tmax-2)
# ax2.set_xlabel("Year",fontsize=20, **hfont)
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
# # y-axis
# ax1.set_ylabel("Catch $tons$", rotation=90, labelpad=5, fontsize=20, **hfont)
# ax2.set_ylabel("Mantle length $cm$", rotation=270, color='silver', labelpad=22, fontsize=20, **hfont)
# plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# # legend
# plt.legend([line1, line2, line3], ["Prediction", "Data", "Mantle length"], fontsize= 11)
# # save and show
# # fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support2MC.png',dpi=300)
# plt.show()
#
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(PrAll[10:-1], meanP[10:-1])
# print ("st error:", std_err)
# print("r-squared price:", r_value**2)
# print ("p-value:", scipy.stats.pearsonr(PrAll[10:-1], meanP[10:-1]))
#
# ### catch
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanC[10:-1])
# print ("st error:", std_err)
# print("r-squared catch:", r_value**2)
# print ("p-value:", scipy.stats.pearsonr(VolAll[10:-1], meanC[10:-1]))
