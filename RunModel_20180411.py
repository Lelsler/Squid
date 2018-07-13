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
