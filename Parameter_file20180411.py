#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

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
