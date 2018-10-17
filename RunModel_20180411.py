#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from scipy import stats

#### model wo relationships ###################################################
flag = 0 # 0 = NoR model; 1 = Rmodel

##### Run the model ############################################################
OUT = np.zeros(tau.shape[0])
OUT1 = np.zeros(tau.shape[0])

for i in np.arange(0,tau.shape[0]):
        tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f, R = model(b0, b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag)
        OUT[i]= p_f[i]
        OUT1[i]= C[i]

#### model w relationships ###################################################
flag = 1 # 0 = NoR model; 1 = Rmodel

##### Run the model ############################################################
OUT2 = np.zeros(tau.shape[0])
OUT3 = np.zeros(tau.shape[0])

for i in np.arange(0,tau.shape[0]):
        tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f, R = model(b0, b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag)
        OUT2[i]= p_f[i]
        OUT3[i]= C[i]

##### Load stuff ###############################################################
###! AS NPY
NoRPold= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Old/PY/DATA/ModelNoR_pf_20180520.npy")
NoRCold= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Old/PY/DATA/ModelNoR_C_20180520.npy")

RPold= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Old/PY/DATA/ModelR_pf_20180520.npy")
RCold= np.load("./Dropbox/PhD/Resources/Squid/Squid/CODE/Old/PY/DATA/ModelR_C_20180520.npy")

### Load dataset ###############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Old/PY/R3_data.xlsx', sheetname='Sheet1')
#! load columns
yr = df1['year'] #
pe = df1['pe_MXNiat'] #
pf = df1['pf_MXNiat'] #
ct = df1['C_t'] #
ssh = df1['essh_avg'] #
ml = df1['ML'] #
ys = df1['y_S'] #

df2 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Old/PY/PriceVolDataCorrected.xlsx', sheetname='Sheet1')
# Load columns
VolAll = df2['tons_DM']
PrAll = df2['priceMXNia_DM']

##### plot stuff ###############################################################
fig = plt.figure()
a, = plt.plot(NoRPold, label = "BEM+ old")
b, = plt.plot(RPold, label = "BEM old")
c, = plt.plot(OUT, label = "BEM+ new")
d, = plt.plot(OUT2, label = "BEM new")
e, = plt.plot(PrAll, label = "data", linewidth=3)
# title
plt.title("Predicted and measured price", fontsize= 25)
# x-axis
plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(10,tmax)
plt.xlabel("time",fontsize=20)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
plt.ylabel("Price",fontsize=20)
plt.ylim(0,5e4)
# legend
plt.legend(handles=[a,b,c,d,e], loc='best')
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Old/PY/tau_newtau_Pf.png',dpi=200)
plt.show()

Pr1 = PrAll[10:26]
Pr2 = NoRPold[10:26]
Pr3 = RPold[10:26]
Pr4 = OUT[10:26]
Pr5 = OUT2[10:26]

slope, intercept, r_value, p_value, std_err = stats.linregress(Pr1, Pr2)
print("r-squared price BEM isotherm:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(Pr1, Pr3)
print("r-squared price BEM+ isotherm:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(Pr1, Pr4)
print("r-squared price BEM SST:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(Pr1, Pr5)
print("r-squared price BEM+ SST:", r_value**2)
print("pearson correlation coefficient and p-value catch BEM+ SST:", stats.pearsonr(Pr1,Pr5))


##### plot stuff ###############################################################
fig = plt.figure()
a, = plt.plot(NoRCold, label = "BEM+ old")
b, = plt.plot(RCold, label = "BEM old")
c, = plt.plot(OUT1, label = "BEM+ new")
d, = plt.plot(OUT3, label = "BEM new")
e, = plt.plot(VolAll, label = "data", linewidth = 3)
# title
plt.title("Predicted and measured catch", fontsize= 25)
# x-axis
plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(10,tmax)
plt.xlabel("time",fontsize=20)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
plt.ylabel("catch",fontsize=20)
# legend
plt.legend(handles=[a,b,c,d,e], loc='best')
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Old/PY/tau_newtau_C.png',dpi=200)
plt.show()

Vol1 = VolAll[10:26]
Vol2 = NoRCold[10:26]
Vol3 = RCold[10:26]
Vol4 = OUT1[10:26]
Vol5 = OUT3[10:26]

slope, intercept, r_value, p_value, std_err = stats.linregress(Vol1, Vol2)
print("r-squared catch BEM isotherm:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(Vol1, Vol3)
print("r-squared catch BEM+ isotherm:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(Vol1, Vol4)
print("r-squared catch BEM SST:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(Vol1, Vol5)
print("r-squared catch BEM+ SST:", r_value**2)
print("pearson correlation coefficient and p-value catch BEM+ SST:", stats.pearsonr(Vol1,Vol5))
