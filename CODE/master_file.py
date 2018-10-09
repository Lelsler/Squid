#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

### Parameters #################################################################
# scales: tons, years, MXNia, hours, trips
tmax = 30 # model run, years
b1 = 41.750 # isotherm depth
b2 = -5.696 # isotherm depth
b3 = 16.397 # isotherm depth
n1 = -22.239 # ML, slope
n2 = 49.811 # ML, intersect
l1 = -0.0028 # q, slope
l2 = 0.1667 # q, intersect
a1 = 1/3.4E7 # proportion of migrating squid, where 3.4E7 max(e^(tau-b1))
g = 1.4 # population growth rate
K = 1208770 # carrying capacity in t
m = 5492603.58 # cost per unit of transport all boats, MXN/trip
f = 40 # l of fuel per trip
B_h = 7.203 # hours per fisher
B_f = 2 # fisher per panga
h1 = 2E-10 # scale E
h2 = 0.6596 # scale E
gamma = 49200 # maximum demand, t
beta = 0.0736 # slope of demand-price function
c_p = 1776.25 # cost of processing, MXNia/t
w_m = 13355164 # min wage per hour all fleet
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
tau[0] = 42. # temperature
q[0] = 0.01 # squid catchability
y_S[0] = 0.5 # proportion of migrated squid
R_tt[0] = 0.5 # trader cooperation
S[0] = 1208770 # size of the squid population
c_t[0] = m *g # fleet cost of transport
E[0] = 1. # fishing effort
C[0] = 120877 # squid catch
p_e[0] = 164706 # max p_e comtrade
p_f[0] = 15438 # max p_f datamares
R[0] = C[0] *p_f[0] -(c_t[0] + E[0])

################################################################################
###############################  MODEL FILE  ###################################
################################################################################

### Define Model ###############################################################
def model(b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag):
    for t in np.arange(1,tmax):
        # isotherm depth
        tau[t]= b1 +b2 *np.cos(t) + b3 *np.sin(t)
        # mantle length
        ML[t]= n1 *tau[t] + n2
        # catchability
        q[t]= l1 *tau[t] +l2
        # migration of squid
        y_S[t] = a1 *np.exp(tau[t]-b1)
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
###############################  RUN MODEL FILE  ###############################
################################################################################

#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

#### model w/o relationships ###################################################
flag = 0 # 0 = NoR model; 1 = Rmodel

##### Run the model ############################################################
OUT = np.zeros(tau.shape[0])
OUT1 = np.zeros(tau.shape[0])

for i in np.arange(0,tau.shape[0]):
        tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f, R = model(b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag)
        OUT[i]= p_f[i]
        OUT1[i]= C[i]

##### Save stuff ###############################################################
###! AS CSV
# np.savetxt("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/ModelR_20180327_p_f.csv", OUT, delimiter=",")
###! AS NPY
# np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/ModelR_CTest.npy", OUT1)

################################################################################
#################################  PLOT FILE  ##################################
################################################################################

### Load data ##################################################################
###! Model outputs
R_C1 = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/ModelR_C_20180411.npy")
R_pf1 = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/ModelR_pf_20180411.npy")

NoR_C1 = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/ModelNoR_C_20180411.npy")
NoR_pf1 = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/ModelNoR_pf_20180411.npy")

# Exclude first data point
R_C = R_C1[1:]
R_pf = R_pf1[1:]

NoR_C = NoR_C1[1:]
NoR_pf = NoR_pf1[1:]

###! Load data
df1 = pd.read_excel('./Dropbox/PhD/Resources/P2/Squid/Laura/PriceVolDataCorrected.xlsx', sheetname='Sheet1')

# Load columns
VolAll = df1['tons_DM']
VolEtal = df1['tons_DM_etal']
VolSR = df1['tons_DM_SR']

PrAll = df1['priceMXNia_DM']
PrEtal = df1['priceMXNia_DM_etal']
PrSR = df1['priceMXNia_DM_SR']

#### PLOT ######################################################################
###! Scatter plot
x = range(100000)
y = range(0,4)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(R_C, R_pf, s=30, c='b', marker="s", label='model with Rtt')
ax1.scatter(NoR_C, NoR_pf, s=30, c='y', marker="o", label='model w/o Rtt')
ax1.scatter(VolAll, PrEtal, s=30, c='g', marker="o", label='Others data')
ax1.scatter(VolAll, PrSR, s=30, c='r', marker="s", label='SR data')
plt.title("Price/catch: model and data", fontsize= 25)
plt.xlabel("Catch in t",fontsize=20)
plt.ylabel("Price for fishers in MXN",fontsize=20)
plt.legend(loc="best", fontsize=10);
# fig.savefig('./Dropbox/PhD/Resources/P2/Squid/CODE/PY/FIGS/R1_20180411.png',dpi=200)
plt.show()

###! Time series plot
fig = plt.figure()
a, = plt.plot(R_C, label = "R catch")
b, = plt.plot(NoR_C, label = "NoR catch")
e, = plt.plot(VolAll, label = "data catch")
c, = plt.plot(R_pf, label = "R pf")
d, = plt.plot(NoR_pf, label = "NoR pf")
f, = plt.plot(PrAll, label = "data price")
plt.xlim(2,len(R_C)-2)
#plt.ylim(0,3)
plt.xlabel("yr",fontsize=20)
plt.ylabel("variables",fontsize=20)
plt.legend(handles=[a,b,c,d,e,f], loc='best')
plt.title("Test", fontsize= 25)
#fig.savefig('./Dropbox/PhD/Deliverables/3_March/Week1/CpfPred_Rtt.png',dpi=200)
plt.show()
