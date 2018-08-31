### Load packages ##############################################################
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

### Load data ##################################################################
###! Model outputs
R_C1 = np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelR_C_20180411.npy")
R_P1 = np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelR_pf_20180411.npy")
meanNoR_C =np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanC.npy")
meanNoR_P =np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanP.npy")

NoR_C1 = np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelNoR_C_20180411.npy")
NoR_P1 = np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelNoR_pf_20180411.npy")
meanR_C =np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_R_meanC.npy")
meanR_P =np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_R_meanP.npy")

# Exclude first data point
R_C = R_C1[:-3]
R_P = R_P1[:-3]

NoR_C = NoR_C1[:-3]
NoR_P = NoR_P1[:-3]

###! Load data
df1 = pd.read_excel('/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/PriceVolDataCorrected.xlsx', sheetname='Sheet1')

# Load columns
VolAll = df1['tons_DM']
VolEtal = df1['tons_DM_etal']
VolSR = df1['tons_DM_SR']

PrAll = df1['priceMXNia_DM']
PrEtal = df1['priceMXNia_DM_etal']
PrSR = df1['priceMXNia_DM_SR']

#### PLOT ######################################################################
hfont = {'fontname':'Helvetica'}

###! Scatter plot
x = range(100000)
y = range(0,4)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(meanR_C, meanR_P, s=30, color="orange", marker="o", label='BEM+')
ax1.scatter(meanNoR_C, meanNoR_P, s=30, color="steelblue", marker="o", label='BEM')
ax1.scatter(VolSR, PrSR, s=30, color="indianred", marker="s", label='SR data')
ax1.scatter(VolAll, PrAll, s=30, color="maroon", marker="s", label='All offices data')
# both axis
plt.tick_params(axis='both', which='major', labelsize=12)
# x-axis
plt.xlabel("Catch $tons$",fontsize=22, **hfont)
plt.xlim(1,1E5)
# y-axis
plt.ylabel("Price for fishers $MXN$",fontsize=22, **hfont)
plt.ylim(1,)
# legend
plt.legend(loc="best", fontsize=14);
# save &show stuff
# fig.savefig('/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/FIGS/R1_20180411.png',dpi=200)
plt.show()

###! Time series plot
fig = plt.figure()
a, = plt.plot(R_C, label = "R catch")
b, = plt.plot(NoR_C, label = "NoR catch")
e, = plt.plot(VolAll, label = "data catch")
c, = plt.plot(R_P, label = "R pf")
d, = plt.plot(NoR_P, label = "NoR pf")
f, = plt.plot(PrAll, label = "data price")
plt.xlim(2,len(R_C)-2)
#plt.ylim(0,3)
plt.xlabel("yr",fontsize=20)
plt.ylabel("variables",fontsize=20)
plt.legend(handles=[a,b,c,d,e,f], loc='best')
plt.title("Test", fontsize= 25)
plt.show()

### CALCULATE r squared ########################################################
### prep data
A = PrAll[:-1]
B = R_P[:-1]
C = NoR_P[:-1]
D = meanR_P[:-1]
E = meanNoR_P[:-1]
# F = newR_P[:-1]
# G = newNoR_P[:-1]

H = VolAll[:-1]
I = R_C[:-1]
J = NoR_C[:-1]
K = meanR_C[:-1]
L = meanNoR_C[:-1]
# M = newR_C[:-1]
# N = newNoR_C[:-1]


### price for fishers
# initial parameter value simulation
slope, intercept, r_value, p_value, std_err = stats.linregress(A,B)
print("r-squared price BEM+:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(A,C)
print("r-squared price BEM:", r_value**2)

# mean from Monte Carlo simulation
slope, intercept, r_value, p_value, std_err = stats.linregress(A,D)
print("r-squared price BEM+:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(A,E)
print("r-squared price BEM:", r_value**2)

# new parameter value simulation (SSH, r&K)
# slope, intercept, r_value, p_value, std_err = stats.linregress(A,F)
# print("r-squared price BEM+:", r_value**2)
#
# slope, intercept, r_value, p_value, std_err = stats.linregress(A,G)
# print("r-squared price BEM:", r_value**2)

### catch
# initial parameter value simulation
slope, intercept, r_value, p_value, std_err = stats.linregress(H,I)
print("r-squared catch BEM+:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(H,J)
print("r-squared catch BEM:", r_value**2)

# mean from Monte Carlo simulation
slope, intercept, r_value, p_value, std_err = stats.linregress(H,K)
print("r-squared catch BEM+:", r_value**2)

slope, intercept, r_value, p_value, std_err = stats.linregress(H,L)
print("r-squared catch BEM:", r_value**2)

# new parameter value simulation (SSH, r&K)
# slope, intercept, r_value, p_value, std_err = stats.linregress(H,M)
# print("r-squared catch BEM+:", r_value**2)
#
# slope, intercept, r_value, p_value, std_err = stats.linregress(H,N)
# print("r-squared catch BEM:", r_value**2)
