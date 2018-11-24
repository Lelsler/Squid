#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy
from pandas import *

#################### CHOICES ###################################################
flag = 2 # 0 = BEM; 1 = MLM, # 2 = BLM

# Standard set-up for model runs: x1110110
tmax = 27 # model run, years

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
def model(b0, b1, b2, b3, l1, l2, qc, a1, d, f, g, K, h1, h2, gamma, beta, c_p, c_t, flag):
    for t in np.arange(1,tmax):
        tau[t]= b0 +b1 *(t+2015) +b2 *np.cos(t+2015) + b3 *np.sin(t+2015)

        # mantle length and catchability
        if ml[t] == 1:
            q[t]= l1 *tau[t] +l2
        else:
            ML[t]= ml[t]
            q[t]= 0.0018 *ML[t] - 0.0318

        if q[t] > 0.1:
            q[t] = 0.1
            print "q high"
        elif q[t] < 0:
            q[t] = 0
            print "q low"

        # migration of squid
        if ys[t] == 1: # run w/o data
            y_S[t] = a1 *np.exp(tau[t]-(b0 +b1*(t+2015))) # sst trend
        else:
            y_S[t]= ys[t] # run with data

        if y_S[t] > 1:
            y_S[t] = 1
            print "yS high"
        elif y_S[t] < 0:
            y_S[t] = 0
            print "yS low"

        # trader cooperation
        R_tt[t]= f+ np.exp(-d* y_S[t])

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
        p_e[t] = gamma* (C[t])**(-beta)
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
###########################  RUN MODEL FILE  ###################################
################################################################################

##### Initiate arrays ##########################################################
sim = np.arange(0,100) # number of simulations
x = np.zeros(12) # set array to save parameters
par = np.zeros((sim.shape[0],x.shape[0])) # matrix to save parameter values of each simulation
cat = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save catches in each time period of each simulation
pri = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save prices in each time period of each simulation

### extra variables to monitor
tem = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save tau in each time period of each simulation
mig = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save migrate squid in each time period of each simulation
cco = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save catchability in each time period of each simulation
pop = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save squid population in each time period of each simulation
eff = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save effort in each time period of each simulation
mar = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save market prices in each time period of each simulation

gap = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save revenue gap
rvf = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save income fishers
rvt = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save income traders
rva = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save revenue fishery


##### Run the model ############################################################
for j in range(0,sim.shape[0]): # draw randomly a float in the range of values for each parameter
    qc = np.random.uniform(0.01, 0.5)
    d = np.random.uniform(0.5, 1.5)
    g = np.random.uniform(0, 2.9)
    gamma = np.random.uniform(20000, 51000)
    beta = np.random.uniform(0.01, 0.1)
    c_p = np.random.uniform(1000, 2148)
    c_t = np.random.uniform(50907027, 212300758)

    b0 = np.random.uniform(-5.15, -27.83) #SST
    b1 = np.random.uniform(0.026, 0.014) #SST
    b2 = np.random.uniform(6.859, 6.699) #SST
    b3 = np.random.uniform(0.171, 0.011) #SST
    x = [b0, b1, b2, b3, qc, d, g, gamma, beta, c_p, c_t, w_m]

    par[j] = x

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

    for i in np.arange(1,tmax):
            tau, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA, G = model(b0, b1, b2, b3, l1, l2, qc, a1, d, f, g, K, h1, h2, gamma, beta, c_p, c_t, flag)
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
            rva[j,i] = OUT12[i]

lowC = np.zeros(y.shape[0]) # initiate variables for 95% confidence interval
highC = np.zeros(y.shape[0])
meanC = np.zeros(y.shape[0])
lowP = np.zeros(y.shape[0])
highP = np.zeros(y.shape[0])
meanP = np.zeros(y.shape[0])

for h in range(0,y.shape[0]): # calculate the 95% confidence interval
    z = cat[:,h]
    lowC[h] = np.nanmean(z) - ((1.96 * np.nanstd(z))/np.sqrt(np.count_nonzero(~np.isnan(z))))
    highC[h] = np.nanmean(z) + ((1.96 * np.nanstd(z))/np.sqrt(np.count_nonzero(~np.isnan(z))))
    meanC[h] = np.nanmean(z)
    zeta = pri[:,h]
    lowP[h] = np.nanmean(zeta) - ((1.96 * np.nanstd(zeta))/np.sqrt(np.count_nonzero(~np.isnan(zeta))))
    highP[h] = np.nanmean(zeta) + ((1.96 * np.nanstd(zeta))/np.sqrt(np.count_nonzero(~np.isnan(zeta))))
    meanP[h] = np.nanmean(zeta)


#### Save data  ###############################################################
# if flag == 0:
#     print "bem save"
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_lowC.npy", lowC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_highC.npy", highC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_meanC.npy", meanC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_lowP.npy", lowP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_highP.npy", highP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanP.npy", meanP)
# if flag == 1:
#     print "mlm save"
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_lowC.npy", lowC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_highC.npy", highC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_meanC.npy", meanC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_lowP.npy", lowP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_highP.npy", highP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanP.npy", meanP)
# if flag == 2:
#     print "blm save"
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_OR_lowC.npy", lowC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_OR_highC.npy", highC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_OR_meanC.npy", meanC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_OR_lowP.npy", lowP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_OR_highP.npy", highP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_OR_meanP.npy", meanP)

################################################################################
###########################  PLOT FILE  ########################################
################################################################################
#
#
# #### Load data  #R1support2_95_NoR_meanC.npy##############################################################
# NoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesNoR_pf.npy")
# NoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesNoR_C.npy")
# lowNoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_lowP.npy")
# highNoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_highP.npy")
# meanNoR_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanP.npy")
# lowNoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_lowC.npy")
# highNoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_highC.npy")
# meanNoR_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_meanC.npy")
#
# lowR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_lowP.npy")
# highR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_highP.npy")
# meanR_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanP.npy")
# lowR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_lowC.npy")
# highR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_highC.npy")
# meanR_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_meanC.npy")
#
# lowOR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_OR_lowP.npy")
# highOR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_OR_highP.npy")
# meanOR_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_OR_meanP.npy")
# lowOR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_OR_lowC.npy")
# highOR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_OR_highC.npy")
# meanOR_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_OR_meanC.npy")
#
# ### Load dataset ###############################################################
# df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3_data.xlsx', sheetname='Sheet1')
# #! load columns
# yr = df1['year'] #
# pe = df1['pe_MXNiat'] #
# pf = df1['pf_MXNiat'] #
# ct = df1['C_t'] #
# ssh = df1['essh_avg'] #
# ml = df1['ML'] #
# ys = df1['y_S'] #
#
# df2 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/PriceVolDataCorrected.xlsx', sheetname='Sheet1')
# # Load columns
# VolAll = df2['tons_DM'] ## CATCH DATA
# PrAll = df2['priceMXNia_DM'] ## PRICE DATA
#
# ### New max time ###############################################################
# tmax = len(yr)
# x = np.arange(0,len(yr))
#
# ### font ######################################################################
# hfont = {'fontname':'Helvetica'}
#
# #####! PLOT MODEL  #############################################################
# fig = plt.figure()
# # add the first axes using subplot populated with predictions
# ax1 = fig.add_subplot(111)
# line1, = ax1.plot(meanR_P, label = "MLM", color="orange")
# line2, = ax1.plot(meanOR_P, label = "BLM", color="sage")
# line3, = ax1.plot(meanNoR_P, label = "BEM", color="steelblue")
# line4, = ax1.plot(PrAll, label = "data", color = "indianred")
# ax1.fill_between(x, highR_pf, lowR_pf, where = highNoR_pf >= lowNoR_pf, facecolor='orange', alpha= 0.3, zorder = 0)
# ax1.fill_between(x, highNoR_pf, lowNoR_pf, where = highNoR_pf >= lowNoR_pf, facecolor='steelblue', alpha= 0.3, zorder = 0)
# ax1.fill_between(x, highOR_pf, lowOR_pf, where = highOR_pf >= lowOR_pf, facecolor='sage', alpha= 0.3, zorder = 0)
# # add the second axes using subplot with ML
# ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
# line5, = ax2.plot(ml, color="lightgrey")
# # x-axis
# ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax1.set_xlim(10,tmax-2)
# ax1.set_xlabel("Year",fontsize=20, **hfont)
# ax2.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax2.set_xlim(10,tmax-2)
# ax2.set_xlabel("Year",fontsize=20, **hfont)
# # y-axis
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
# ax2.set_ylim(0,140)
# ax1.set_ylabel("Prices for fishers $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
# ax2.set_ylabel("Mantle length $cm$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
# ax2.tick_params(axis='y', colors='lightgrey')
# plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# # legend
# plt.legend([line1, line2, line3, line4], ["MLM", "BLM", "BEM", "Data"], fontsize= 11)
# # save and show
# # fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support1MC.png',dpi=500)
# plt.show()
#
# fig = plt.figure()
# # add the first axes using subplot populated with predictions
# ax1 = fig.add_subplot(111)
# line1, = ax1.plot(meanR_C, label = "MLM", color="orange")
# line2, = ax1.plot(meanOR_C, label = "BLM", color="sage")
# line3, = ax1.plot(meanNoR_C, label = "BEM", color="steelblue")
# line4, = ax1.plot(VolAll, label = "data", color = "indianred")
# ax1.fill_between(x, highR_C, lowR_C, where = highR_C >= lowR_C, facecolor='orange', alpha= 0.3, zorder = 0)
# ax1.fill_between(x, highNoR_C, lowNoR_C, where = highNoR_C >= lowNoR_C, facecolor='steelblue', alpha= 0.3, zorder = 0)
# ax1.fill_between(x, highOR_C, lowOR_C, where = highOR_C >= lowOR_C, facecolor='sage', alpha= 0.3, zorder = 0)
# # add the second axes using subplot with ML
# ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
# line5, = ax2.plot(ml, color="lightgrey")
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
# ax1.set_ylim(0,3E5)
# ax2.set_ylabel("Mantle length $cm$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
# ax2.set_ylim(0,140)
# ax2.tick_params(axis='y', colors='lightgrey')
# plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# # legend
# plt.legend([line1, line2, line3, line4], ["MLM", "BLM", "BEM", "Data"], loc=1, fontsize= 11)
# # save and show
# # fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support2MC.png',dpi=200)
# plt.show()
#
# ### CALCULATE r squared ########################################################
# ### price for fishers
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(PrAll[10:-1], meanR_P[10:-1])
# print("r-squared price MLM:", r_value**2)
# scipy.stats.pearsonr(PrAll[10:-1], meanR_P[10:-1])
#
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(PrAll[10:-1], meanNoR_P[10:-1])
# print("r-squared price BEM:", r_value**2)
# scipy.stats.pearsonr(PrAll[10:-1], meanNoR_P[10:-1])
#
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(PrAll[10:-1], meanOR_P[10:-1])
# print("r-squared price BLM:", r_value**2)
# scipy.stats.pearsonr(PrAll[10:-1], meanOR_P[10:-1])
#
# ### catch
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanR_C[10:-1])
# print("r-squared catch MLM:", r_value**2)
# scipy.stats.pearsonr(VolAll[10:-1], meanR_C[10:-1])
#
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanNoR_C[10:-1])
# print("r-squared catch BEM:", r_value**2)
# scipy.stats.pearsonr(VolAll[10:-1], meanNoR_C[10:-1])
#
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanOR_C[10:-1])
# print("r-squared catch BLM:", r_value**2)
# scipy.stats.pearsonr(VolAll[10:-1], meanOR_C[10:-1])

################################################################################
###########################  PLOT TEST - direclty plots from model output  #####
################################################################################

### Load dataset ###############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3_data.xlsx', sheetname='Sheet1')
#! load columns
yr = df1['year'] #
pe = df1['pe_MXNiat'] #
pf = df1['pf_MXNiat'] #
ct = df1['C_t'] #
ssh = df1['essh_avg'] #
ml = df1['ML'] #
ys = df1['y_S'] #

df2 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/PriceVolDataCorrected.xlsx', sheetname='Sheet1')
# Load columns
VolAll = df2['tons_DM'] ## CATCH DATA
PrAll = df2['priceMXNia_DM'] ## PRICE DATA
x = np.arange(0,len(yr))

hfont = {'fontname':'Helvetica'}

fig = plt.figure()
# add the first axes using subplot populated with predictions
ax1 = fig.add_subplot(111)
line1, = ax1.plot(meanP, label = "MLM", color="orange")
line2, = ax1.plot(PrAll, label = "data", color = "indianred")
ax1.fill_between(x, highP, lowP, where = highP >= lowP, facecolor='orange', alpha= 0.3, zorder = 0)
# add the second axes using subplot with ML
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line3, = ax2.plot(ml, color="lightgrey")
# x-axis
ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
ax1.set_xlim(10,tmax-2)
ax1.set_xlabel("Year",fontsize=20, **hfont)
ax2.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
ax2.set_xlim(10,tmax-2)
ax2.set_xlabel("Year",fontsize=20, **hfont)
# y-axis
ax1.set_ylabel("Prices for fishers $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax2.set_ylabel("Mantle length $cm$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
# legend
plt.legend([line1, line2, line3], ["Prediction", "Data", "Mantle length"], fontsize= 11)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support1MC.png',dpi=500)
plt.show()


fig = plt.figure()
# add the first axes using subplot populated with predictions
ax1 = fig.add_subplot(111)
line1, = ax1.plot(meanC, label = "MLM", color="orange")
line2, = ax1.plot(VolAll, label = "data", color = "indianred")
ax1.fill_between(x, highC, lowC, where = highC >= lowC, facecolor='orange', alpha= 0.3, zorder = 0)
# add the second axes using subplot with ML
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line3, = ax2.plot(ml, color="lightgrey")
# x-axis
ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
ax1.set_xlim(10,tmax-2)
ax1.set_xlabel("Year",fontsize=20, **hfont)
ax2.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
ax2.set_xlim(10,tmax-2)
ax2.set_xlabel("Year",fontsize=20, **hfont)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
# y-axis
ax1.set_ylabel("Catch $tons$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax2.set_ylabel("Mantle length $cm$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# legend
plt.legend([line1, line2, line3], ["Prediction", "Data", "Mantle length"], fontsize= 11)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support2MC.png',dpi=200)
plt.show()

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(PrAll[10:-1], meanP[10:-1])
print ("st error:", std_err)
print("r-squared price:", r_value**2)
print ("p-value:", scipy.stats.pearsonr(PrAll[10:-1], meanP[10:-1]))

### catch
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanC[10:-1])
print ("st error:", std_err)
print("r-squared catch:", r_value**2)
print ("p-value:", scipy.stats.pearsonr(VolAll[10:-1], meanC[10:-1]))
