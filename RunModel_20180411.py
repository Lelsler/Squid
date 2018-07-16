#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

#### model w/o relationships ###################################################
flag = 1 # 0 = NoR model; 1 = Rmodel

##### Run the model ############################################################
OUT = np.zeros(tau.shape[0])
OUT1 = np.zeros(tau.shape[0])

for i in np.arange(0,tau.shape[0]):
        tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f, R = model(b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag)
        OUT[i]= p_f[i]
        OUT1[i]= C[i]

##### Save stuff ###############################################################
###! AS NPY
if flag == 0:
    np.save("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelNoR_pf_20180411.npy", OUT)
    np.save("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelNoR_C_20180411.npy", OUT1)
    print "model without relationships"

if flag == 1:
    np.save("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelR_pf_20180411.npy", OUT)
    np.save("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelR_C_20180411.npy", OUT1)
    print "model with relationships"
