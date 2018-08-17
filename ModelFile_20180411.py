#### Import packages ###########################################################
import scipy.stats as st
from pandas import *

#### Model w/o relationships ###################################################
flag = 1 # 0 = NoR model; 1 = Rmodel

#### Load dataset  #############################################################
df1 = pd.read_excel('./DATA/R3_data.xlsx', sheet_name='Sheet1')
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
            print("yS high")
        elif y_S[t] < 0:
            y_S[t] = 0
            print("yS low")

        # trader cooperation
        R_tt[t] = (1-y_S[t])
        # squid population
        S[t] = S[t-1] +g *S[t-1] *(1- (S[t-1]/K)) - q[t-1] *E[t-1] *S[t-1]
        # cost of transport
        c_t[t]= m *f # I decided to use fixed costs over migration, that equally well/better predicted catches over m* (y_S[t]); (source: LabNotesSquid, April 11)
        # fishing effort
        Escal[t] = E[t-1] + p_f[t-1] *q[t-1] *E[t-1] *S[t-1] -c_t[t-1] *(E[t-1]/(B_h+B_f)) # c_t is per trip so we need to upscale E hr > fisher > trip
        # fishing effort scaled
        E[t] = h1 *Escal[t] + h2 # Escal €[-3,10E+09; 1,60E+09]
        if E[t] > 1:
            E[t] = 1
            print("E high")
        elif E[t] < 0:
            E[t] = 0
            print("E low")

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

        print(t, tau[t], ML[t], q[t], y_S[t], S[t], c_t[t], E[t], C[t], p_e[t], p_f[t], R[t])
    return tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f, R
