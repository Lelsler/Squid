#### Laura Elsler: February 2019
#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy
from pandas import *

#### Load dataset  #############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_all_plot_data.xlsx', sheetname='Sheet1')
#! load columns
yr = df1['year'] # year
cat = df1['C'] # catches total from Tim data 1991-2014
pg = df1['pf_GY'] # prices for fishers in Guaymas
pr = df1['pf_rm'] # prices for fishers in all other offices except SR, GY
ps = df1['pf_SR'] # prices for fishers in Santa Rosalia
ys = df1['y_S'] # proportion of migrated squid
ml = df1['ML'] # mantle length
sst = df1['sst_anom'] # sea surface temperature anomaly
cg = df1['C_GY'] # catches from datamares in Guaymas
cr = df1['C_RM'] # catches from datamares in all other offices except SR, GY
cs = df1['C_SR'] # catches from datamares in Santa Rosalia
ca = df1['C_AL'] # catches from datamares summed over all offices

### Plot files #################################################################
hfont = {'fontname':'Helvetica'}

### catches
fig = plt.figure()
# add the first axes using subplot
ax1 = fig.add_subplot(111)
line1, = ax1.plot(ca, label = "Total", color="orange")
line2, = ax1.plot(cg, label = "Guaymas", color = "indianred")
line3, = ax1.plot(cs, label = "Santa Rosalia", color = "royalblue")
# x-axis
ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax1.set_xlim(10,tmax-2)
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("Catches $tons$", rotation=90, labelpad=5, fontsize=20, **hfont)
# legend
plt.legend([line1, line2, line3], ["Total", "Guaymas", "Santa Rosalia"], fontsize= 11)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_catch.png',dpi=500)
plt.show()

### prices
fig = plt.figure()
# add the first axes using subplot
ax1 = fig.add_subplot(111)
line1, = ax1.plot(pg, label = "Guaymas", color = "indianred")
line2, = ax1.plot(ps, label = "Santa Rosalia", color = "royalblue")
line3, = ax1.plot(pr, label = "Other offices", color="orange")
# x-axis
ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax1.set_xlim(10,tmax-2)
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("Price for fishers $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
# legend
plt.legend([line1, line2, line3], ["Guaymas", "Santa Rosalia", "Other offices"], fontsize= 11, loc="best")
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_price.png',dpi=500)
plt.show()

### SST anomaly
fig = plt.figure()
# add the first axes using subplot
ax1 = fig.add_subplot(111)
line1, = ax1.plot(sst, label = "SST anomaly", color = "indianred", linewidth=3)
# x-axis
ax1.set_xticklabels(np.arange(2001,2016,2), rotation=45, fontsize= 12)
# ax1.set_xlim(10,tmax-2)
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("SST anomaly $^\circ$C", rotation=90, labelpad=5, fontsize=20, **hfont)
# legend
plt.legend([line1], ["SST anomaly"], fontsize= 11, loc="best")
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_sst.png',dpi=500)
plt.show()
