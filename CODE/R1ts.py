#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy.stats as st
from pandas import *

#### Model w/o relationships ###################################################
flag = 1 # 0 = NoR model; 1 = Rmodel

### Parameters #################################################################
# scales: tons, years, MXNia, hours, trips
tmax = 27 # model run, years
b1 = 41.750 # isotherm depth
b2 = -5.696 # isotherm depth
b3 = 16.397 # isotherm depth
n1 = -22.239 # ML, slope
n2 = 49.811 # ML, intersect
l1 = -0.0028 # q, slope
l2 = 0.1667 # q, intersect
a1 = 1/3.4E7 # proportion of migrating squid, where 3.4E7 max(e^(tau-b1))
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
flag = 3 # this gives an error if its not been changed previously to the right model

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
tmax = len(y)

### Define Model ###############################################################
def model(b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag):
    for t in np.arange(1,tmax):
        # isotherm depth
        tau[t]= b1 +b2 *np.cos(t) + b3 *np.sin(t)
        # mantle length and catchability
        if ml[t] == 1:
            q[t]= l1 *tau[t] +l2
        else:
            ML[t]= ml[t]
            q[t]= 0.0018 *ML[t] - 0.0318

        # migration of squid
        if ys[t] == 1:
            y_S[t] = a1 *np.exp(tau[t]-b1)
        else:
            y_S[t]= ys[t]
        if y_S[t] > 1:
            y_S[t] = 1
            print "yS high"
        elif y_S[t] < 0:
            y_S[t] = 0
            print "yS low"

        # trader cooperation
        R_tt[t] = (1-y_S[t])
        # squid population
        S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]
        # cost of transport
        c_t[t]= m *f # I decided to use fixed costs over migration, that equally well/better predicted catches over m* (y_S[t]); (source: LabNotesSquid, April 11)
        # fishing effort
        Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -c_t[t-1] *(E[t-1]/(B_h*B_f)) # c_t is per trip so we need to upscale E hr > fisher > trip
        # fishing effort scaled
        E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]
        if E[t] > 1:
            E[t] = 1
            print "E high"
        elif E[t] < 0:
            E[t] = 0
            print "E low"

        # catch
        C[t] = q[t] *E[t] *S[t]
        # export price
        p_e[t] = gamma* (C[t])**(-beta)

        #### switch between models ####
        if flag == 0:
            # price for fishers
            p_f[t] = p_e[t] -c_p
        if flag == 1:
            # minimum wage
            p_min[t]= (E[t] *w_m)/C[t]
            # price for fishers
            p_f[t] = (p_e[t] -c_p) *(1-R_tt[t]) +R_tt[t] *p_min[t]

        # revenue of fishers
        R[t] = C[t] *p_f[t] - c_t[t-1] *(E[t-1]/(B_h+B_f))

        print t, tau[t], ML[t], q[t], y_S[t], S[t], c_t[t], E[t], C[t], p_e[t], p_f[t], R[t]
        return tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f, R

################################################################################
###########################  RUN MODEL FILE  ###################################
################################################################################

##### Initiate arrays ##########################################################
sim = np.arange(0,100) # number of simulations
x = np.zeros(12) # set array to save parameters
par = np.zeros((sim.shape[0],x.shape[0])) # matrix to save parameter values of each simulation
cat = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save catches in each time period of each simulation
pri = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save prices in each time period of each simulation

##### Run the model ############################################################
for j in range(0,sim.shape[0]): # draw randomly a float in the range of values for each parameter
    # a1 = np.random.uniform(2E-08, 3E-08) # parameter ranges
    # a2 = np.random.uniform(0.2, 9E-16)
    b1 = np.random.uniform(38.750, 42.1)
    b2 = np.random.uniform(-3.987, -6.9)
    b3 = np.random.uniform(11.478, 16.4)
    beta = np.random.uniform(0.01, 0.1)
    c_p = np.random.uniform(1000, 2148)
    g = np.random.uniform(0, 2.9)
    gamma = np.random.uniform(20000, 51000)
    # l1 = np.random.uniform(-0.0005, -0.0122)
    # l2 = np.random.uniform(0.0317, 0.7927)
    m = np.random.uniform(2368793, 8450159)
    w_m = np.random.uniform(11956952, 28108539)

    x = [a1, b1, b2, b3, beta, c_p, g, gamma, l1, l2, m , w_m]
    par[j] = x

    OUT = np.zeros(tau.shape[0])
    OUT1 = np.zeros(tau.shape[0])

    for i in np.arange(1,tmax):
            tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f, R = model(b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag)
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
# np.save("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanC.npy", meanC)

################################################################################
###########################  PLOT FILE  ########################################
################################################################################

#### Load data  ###############################################################
NoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesNoR_pf.npy")
NoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesNoR_C.npy")
highNoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_highP.npy")
highNoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_highC.npy")
lowNoR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_lowP.npy")
lowNoR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_NoR_lowC.npy")
meanNoR_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanC.npy")
meanNoR_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanP.npy")

R_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesR_pf.npy")
R_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/timeSeriesR_C.npy")
highR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_highP.npy")
highR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_highC.npy")
lowR_pf = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_lowP.npy")
lowR_C = np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support2_95_R_lowC.npy")
meanR_C =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanC.npy")
meanR_P =np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R1support1_95_R_meanP.npy")

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

### New max time ###############################################################
tmax = len(yr)
x = np.arange(0,len(yr))

### font ######################################################################
hfont = {'fontname':'Helvetica'}

#####! PLOT MODEL  #############################################################
fig = plt.figure()
a, = plt.plot(meanR_P, label = "MLM", color="orange")
b, = plt.plot(meanNoR_P, label = "BEM", color="steelblue")
c, = plt.plot(PrAll, label = "data", color = "indianred")
plt.fill_between(x, highR_pf, lowR_pf, where = highNoR_pf >= lowNoR_pf, facecolor='orange', alpha= 0.3, zorder = 0)
plt.fill_between(x, highNoR_pf, lowNoR_pf, where = highNoR_pf >= lowNoR_pf, facecolor='steelblue', alpha= 0.3, zorder = 0)
# plt.title("Predicted and measured price for fishers [MXN]", fontsize= 25)
# x-axis
plt.xticks(np.arange(len(yr)), yr, rotation=45, fontsize= 12)
plt.xlim(10,tmax-2)
plt.xlabel("Year",fontsize=22, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
plt.ylabel("Price for fishers $MXN$",fontsize=22, **hfont)
plt.legend(handles=[a,b,c], loc='best', fontsize= 14)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support1MC.png',dpi=500)
plt.show()

fig = plt.figure()
a, = plt.plot(meanR_C, label = "MLM", color="orange")
b, = plt.plot(meanNoR_C, label = "BEM", color="steelblue")
c, = plt.plot(VolAll, label = "data", color= "indianred")
plt.fill_between(x, highR_C, lowR_C, where = highNoR_C >= lowNoR_C, facecolor='orange', alpha= 0.3, zorder = 0)
plt.fill_between(x, highNoR_C, lowNoR_C, where = highNoR_C >= lowNoR_C, facecolor='steelblue', alpha= 0.3, zorder = 0)
# title
# plt.title("Predicted and measured catch [t]", fontsize= 25)
# x-axis
plt.xticks(np.arange(len(yr)), yr, rotation=45, fontsize=12)
plt.xlim(10,tmax-2)
plt.xlabel("Year",fontsize=22, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
plt.ylabel("Catch $tons$",fontsize=22, **hfont)
# legend
plt.legend(handles=[a,b,c], loc='best', fontsize=14)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R1_support2MC.png',dpi=200)
plt.show()

### CALCULATE r squared ########################################################
### price for fishers
slope, intercept, r_value, p_value, std_err = stats.linregress(PrAll, meanR_P)
print("r-squared price MLM:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(PrAll, meanNoR_P)
print("r-squared price BEM:", r_value**2)

### catch
slope, intercept, r_value, p_value, std_err = stats.linregress(VolAll, meanR_C)
print("r-squared catch MLM:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(VolAll, meanNoR_C)
print("r-squared catch BEM:", r_value**2)
