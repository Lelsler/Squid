#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy
from pandas import *

#################### CHOICES ###################################################
flag = 0 # 0 = NoR model; 1 = Rmodel, # 2 = in-btw model
# SST or isotherm: parameters, initial condition for tau, and MC parameter ranges
temperature = 0 # iso = 0, SST = 1
cost = 1 # 0 = old, 1 = new
initial = 1 # 0 = old, 1 = new
mantle = 0 # 0 = use mantle, 1 = use tau
coop = 1 # 0 = old, 1 = new
wage = 1 # 0 = old, 1 = new

tmax = 27 # model run, years

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
RF = np.zeros(tmax) # revenue of fishers
RT = np.zeros(tmax) # revenue of traders
RA = np.zeros(tmax) # revenue all fishery
G = np.zeros(tmax) # pay gap between fishers and traders

### Parameters #################################################################
# scales: tons, years, MXNia, hours, trips
if temperature == 0: ## isotherm depth
    b1 = 41.750 # isotherm depth
    b2 = -5.696 # isotherm depth
    b3 = 16.397 # isotherm depth
    l1 = -0.0028 # q, slope # SST l1 = -0.0059 # q, slope
    l2 = 0.1667 # q, intersect # SST l2 = 0.1882 # q, intersect
    qc = 0.1 # q, constant catchability
    a1 = 1/3.4E7 # proportion of migrating squid, where 3.4E7 max(e^(tau-b1))
    tau[0] = 42. # isotherm depth
if temperature == 1: #following parameters fitted to SST
    b0 = -16.49 # SST trend
    b1 = 0.02 # SST trend
    b2 = 6.779 # SST trend
    b3 = 0.091 # SST trend
    l1 = -0.0059 # q, slope
    l2 = 0.1882 # q, intersect
    qc = 0.1 # catchability constant
    a1 = 1/(np.exp(30.823998124274-(b0+b1*(30+2015)))) # migration trigger
    tau[0] = 30. # for SST
if cost == 0: # old
    c_t = 156076110 # cost of fishing
if cost == 1: # new
    c_t = 107291548 # cost of fishing

f = 0 # intercept of trader cooperation
d = 1 # slope of trader cooperation
K = 1208770 # carrying capacity
g = 1.4 # population growth rate
gamma = 49200 # maximum demand
beta = 0.0736 # slope of demand-price function
c_p = 1776.25 # cost of processing
w_m = 13355164 # min wage per hour all fleet

h1 = 2E-10 # scale E
h2 = 0.6596 # scale E
B_h = 7.203 # hours per fisher
B_f = 2 # fisher per panga

### Initial values #############################################################
if initial == 0: ## OLD
    q[0] = 0.01 # squid catchability
    y_S[0] = 0.5 # proportion of migrated squid
    R_tt[0] = 0.5 # trader cooperation
    S[0] = 1208770 # size of the squid population
    E[0] = 1. # fishing effort
    C[0] = 120877 # squid catch
    p_e[0] = 99366 # max p_e comtrade
    p_f[0] = 15438 # max p_f datamares

if initial == 1: ## NEW
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
def model(b1, b2, b3, l1, l2, qc, a1, B_h, B_f, d, f, g, K, h1, h2, gamma, beta, c_p, c_t, w_m, flag):
    for t in np.arange(1,tmax):
        if temperature == 0: # isotherm depth
            tau[t]= b1 +b2 *np.cos(t) + b3 *np.sin(t)
        if temperature == 1: # sst trend
            tau[t]= b0 +b1 *(t+2015) +b2 *np.cos(t+2015) + b3 *np.sin(t+2015)

        # mantle length and catchability
        if mantle == 0:
            if ml[t] == 1:
                q[t]= l1 *tau[t] +l2
            else:
                ML[t]= ml[t]
                q[t]= 0.0018 *ML[t] - 0.0318
        if mantle == 1:
            q[t]= l1 *tau[t] +l2

        if q[t] > 0.1:
            q[t] = 0.1
            print "q high"
        elif q[t] < 0:
            q[t] = 0
            print "q low"

        # migration of squid
        if mantle == 0: # USE DATA INPUT?
            if ys[t] == 1: # run w/o data
                if temperature == 0: # ISO
                    y_S[t] = a1 *np.exp(tau[t]-b1)
                if temperature == 1: # SST
                    y_S[t] = a1 *np.exp(tau[t]-(b0 +b1*(t+2015))) # sst trend
            else:
                y_S[t]= ys[t] # run with data
        if mantle == 1: # use simulation input
            if temperature == 0: # ISO
                y_S[t] = a1 *np.exp(tau[t]-b1)
            if temperature == 1: # SST
                y_S[t] = a1 *np.exp(tau[t]-(b0 +b1*(t+2015))) # sst trend

        if y_S[t] > 1:
            y_S[t] = 1
            print "yS high"
        elif y_S[t] < 0:
            y_S[t] = 0
            print "yS low"

        # trader cooperation
        if coop == 0: # LINEAR
            R_tt[t] = (1-y_S[t])
        if coop == 1: # EXPONENTIAL
            R_tt[t]= f+ np.exp(-d* y_S[t])

        #### switch between models ####
        if flag == 0: # squid population BEM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - qc *E[t-1] *S[t-1]
        if flag >= 1: # squid population MLM
            S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]

        #### switch between models ####
        if flag == 0: # effort BEM
            Escal[t] = E[t-1] + p_f[t-1] *qc *E[t-1] *S[t-1] -c_t *(E[t-1])
        if flag >= 1: # effort MLM
            Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -c_t *(E[t-1])

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
        if flag == 2: # BEM with tau
            p_f[t] = p_e[t] -c_p    # price for fishers
        if flag == 1: #MLM
            if wage == 0: # minimum wage old
                p_min[t]= (E[t] *w_m)/C[t]
            if wage == 1: # minimum wage new
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

##### Initiate arrays ##########################################################
sim = np.arange(0,100) # number of simulations
if temperature == 0: # ISO
    x = np.zeros(11) # set array to save parameters
if temperature == 1: # SST
    x = np.zeros(12) # set array to save parameters
par = np.zeros((sim.shape[0],x.shape[0])) # matrix to save parameter values of each simulation
cat = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save catches in each time period of each simulation
pri = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save prices in each time period of each simulation

##### Run the model ############################################################
for j in range(0,sim.shape[0]): # draw randomly a float in the range of values for each parameter
    qc = np.random.uniform(0.01, 0.5)
    d = np.random.uniform(0.5, 1.5)
    g = np.random.uniform(0, 2.9)
    gamma = np.random.uniform(20000, 51000)
    beta = np.random.uniform(0.01, 0.1)
    c_p = np.random.uniform(1000, 2148)
    c_t = np.random.uniform(50907027, 212300758)
    w_m = np.random.uniform(11956952, 28108539)

    if temperature == 0: # ISO
        b1 = np.random.uniform(38.750, 42.1) # isotherm
        b2 = np.random.uniform(-3.987, -6.9) # isotherm
        b3 = np.random.uniform(11.478, 16.4) # isotherm
        x = [b1, b2, b3, qc, d, g, gamma, beta, c_p, c_t, w_m]

    if temperature == 1: # SST
        b0 = np.random.uniform(-5.15, -27.83) #SST
        b1 = np.random.uniform(0.026, 0.014) #SST
        b2 = np.random.uniform(6.859, 6.699) #SST
        b3 = np.random.uniform(0.171, 0.011) #SST
        x = [b0, b1, b2, b3, qc, d, g, gamma, beta, c_p, c_t, w_m]

    par[j] = x

    OUT = np.zeros(tau.shape[0])
    OUT1 = np.zeros(tau.shape[0])

    for i in np.arange(1,tmax):
            tau, ML, q, y_S, R_tt, S, E, C, p_e, p_f, RF, RT, RA, G = model(b1, b2, b3, l1, l2, qc, a1, B_h, B_f, d, f, g, K, h1, h2, gamma, beta, c_p, c_t, w_m, flag)
            OUT[i]= p_f[i]
            OUT1[i]= C[i]
            pri[j,i] = OUT[i]
            cat[j,i] = OUT1[i]


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

################################################################################
###########################  PLOT TEST  ########################################
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
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
# y-axis
ax1.set_ylabel("Prices for fishers $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax2.set_ylabel("Mantle length $cm$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
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


##### Save data  ###############################################################
# if flag == 0:
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_lowC.npy", lowC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_highC.npy", highC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_lowP.npy", lowP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_highP.npy", highP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanP.npy", meanP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanC.npy", meanC)
# if flag == 1:
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_lowC.npy", lowC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_highC.npy", highC)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_lowP.npy", lowP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_highP.npy", highP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanP.npy", meanP)
#     np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanC.npy", meanC)


################################################################################
###########################  PLOT FILE  ########################################
################################################################################
#
#
# #### Load data  ###############################################################
# NoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesNoR_pf.npy")
# NoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesNoR_C.npy")
# highNoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_highP.npy")
# highNoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_highC.npy")
# lowNoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_lowP.npy")
# lowNoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_lowC.npy")
# meanNoR_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanC.npy")
# meanNoR_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanP.npy")
#
# R_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesR_pf.npy")
# R_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesR_C.npy")
# highR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_highP.npy")
# highR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_highC.npy")
# lowR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_lowP.npy")
# lowR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_lowC.npy")
# meanR_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanC.npy")
# meanR_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanP.npy")
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
# line2, = ax1.plot(meanNoR_P, label = "BEM", color="steelblue")
# line3, = ax1.plot(PrAll, label = "data", color = "indianred")
# ax1.fill_between(x, highR_pf, lowR_pf, where = highNoR_pf >= lowNoR_pf, facecolor='orange', alpha= 0.3, zorder = 0)
# ax1.fill_between(x, highNoR_pf, lowNoR_pf, where = highNoR_pf >= lowNoR_pf, facecolor='steelblue', alpha= 0.3, zorder = 0)
# # add the second axes using subplot with ML
# ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
# line4, = ax2.plot(ml, color="lightgrey")
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
# ax1.set_ylabel("Prices for fishers $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
# ax2.set_ylabel("Mantle length $cm$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
# plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# # legend
# plt.legend([line1, line2, line3, line4], ["MLM", "BEM", "Data", "Mantle length"], fontsize= 11)
# # save and show
# # fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support1MC.png',dpi=500)
# plt.show()
#
# fig = plt.figure()
# # add the first axes using subplot populated with predictions
# ax1 = fig.add_subplot(111)
# line1, = ax1.plot(meanR_C, label = "MLM", color="orange")
# line2, = ax1.plot(meanNoR_C, label = "BEM", color="steelblue")
# line3, = ax1.plot(VolAll, label = "data", color = "indianred")
# ax1.fill_between(x, highR_C, lowR_C, where = highNoR_C >= lowNoR_C, facecolor='orange', alpha= 0.3, zorder = 0)
# ax1.fill_between(x, highNoR_C, lowNoR_C, where = highNoR_C >= lowNoR_C, facecolor='steelblue', alpha= 0.3, zorder = 0)
# # add the second axes using subplot with ML
# ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
# line4, = ax2.plot(ml, color="lightgrey")
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
# ax2.set_ylabel("Mantle length $cm$", rotation=270, color='lightgrey', labelpad=22, fontsize=20, **hfont)
# plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
# # legend
# plt.legend([line1, line2, line3, line4], ["MLM", "BEM", "Data", "Mantle length"], fontsize= 11)
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
# ### catch
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanR_C[10:-1])
# print("r-squared catch MLM:", r_value**2)
# scipy.stats.pearsonr(VolAll[10:-1], meanR_C[10:-1])
#
# slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(VolAll[10:-1], meanNoR_C[10:-1])
# print("r-squared catch BEM:", r_value**2)
# scipy.stats.pearsonr(VolAll[10:-1], meanNoR_C[10:-1])
