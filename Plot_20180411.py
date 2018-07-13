#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

##### Load data  ###############################################################
NoR_pf = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/timeSeriesNoR_pf.npy")
NoR_C = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/timeSeriesNoR_C.npy")
highNoR_pf = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support1_95_NoR_highP.npy")
highNoR_C = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support2_95_NoR_highC.npy")
lowNoR_pf = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support1_95_NoR_lowP.npy")
lowNoR_C = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support2_95_NoR_lowC.npy")
meanNoR_C =np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support1_95_NoR_meanC.npy")
meanNoR_P =np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support1_95_NoR_meanP.npy")

R_pf = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/timeSeriesR_pf.npy")
R_C = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/timeSeriesR_C.npy")
highR_pf = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support1_95_R_highP.npy")
highR_C = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support2_95_R_highC.npy")
lowR_pf = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support1_95_R_lowP.npy")
lowR_C = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support2_95_R_lowC.npy")
meanR_C =np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support1_95_R_meanC.npy")
meanR_P =np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/PY/DATA/R1support1_95_R_meanP.npy")

### Load dataset ###############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/P2/Squid/Laura/R3_data.xlsx', sheetname='Sheet1')
#! load columns
yr = df1['year'] #
pe = df1['pe_MXNiat'] #
pf = df1['pf_MXNiat'] #
ct = df1['C_t'] #
ssh = df1['essh_avg'] #
ml = df1['ML'] #
ys = df1['y_S'] #

df2 = pd.read_excel('./Dropbox/PhD/Resources/P2/Squid/Laura/PriceVolDataCorrected.xlsx', sheetname='Sheet1')
# Load columns
VolAll = df2['tons_DM']
PrAll = df2['priceMXNia_DM']

### New max time ###############################################################
tmax = len(yr)
x = np.arange(0,len(yr))

### font ######################################################################
hfont = {'fontname':'Helvetica'}

#####! PLOT MODEL  #############################################################
fig = plt.figure()
a, = plt.plot(meanR_P, label = "BEM+", color="orange")
b, = plt.plot(meanNoR_P, label = "BEM", color="steelblue")
c, = plt.plot(PrAll, label = "data", color = "indianred")
plt.fill_between(x, highR_pf, lowR_pf, where = highNoR_pf >= lowNoR_pf, facecolor='orange', alpha= 0.3, zorder = 0)
plt.fill_between(x, highNoR_pf, lowNoR_pf, where = highNoR_pf >= lowNoR_pf, facecolor='steelblue', alpha= 0.3, zorder = 0)
# plt.title("Predicted and measured price for fishers [MXN]", fontsize= 25)
# x-axis
plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(10,tmax-2)
plt.xlabel("year",fontsize=20, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
plt.ylabel("price for fishers $MXN$",fontsize=20, **hfont)
plt.legend(handles=[a,b,c], loc='best')
# save and show
# fig.savefig('./Dropbox/PhD/Resources/P2/Squid/CODE/PY/FIGS/R1_support1MC.png',dpi=500)
plt.show()

fig = plt.figure()
a, = plt.plot(meanR_C, label = "BEM+", color="orange")
b, = plt.plot(meanNoR_C, label = "BEM", color="steelblue")
c, = plt.plot(VolAll, label = "data", color= "indianred")
plt.fill_between(x, highR_C, lowR_C, where = highNoR_C >= lowNoR_C, facecolor='orange', alpha= 0.3, zorder = 0)
plt.fill_between(x, highNoR_C, lowNoR_C, where = highNoR_C >= lowNoR_C, facecolor='steelblue', alpha= 0.3, zorder = 0)
# title
# plt.title("Predicted and measured catch [t]", fontsize= 25)
# x-axis
plt.xticks(np.arange(len(yr)), yr, rotation=45)
plt.xlim(10,tmax-2)
plt.xlabel("year",fontsize=20, **hfont)
plt.gcf().subplots_adjust(bottom=0.15)
# y-axis
plt.ylabel("catch $tons$",fontsize=20, **hfont)
# legend
# plt.legend(handles=[a,b,c], loc='best')
# save and show
# fig.savefig('./Dropbox/PhD/Resources/P2/Squid/CODE/PY/FIGS/R1_support2MC.png',dpi=200)
plt.show()
