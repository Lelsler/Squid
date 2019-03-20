#### Laura Elsler: February 2019
#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from pandas import *
hfont = {'fontname':'Helvetica'}

#### Load dataset  #############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_all_plot_data.xlsx', sheetname='Sheet1')
#! load columns
yr = df1['year'] # year
cat = df1['C'] # catches total from Tim data 1991-2014
pg = df1['pf_GY'] # prices for fishers in Guaymas
pr = df1['pf_RM'] # prices for fishers in all other offices except SR, GY
ps = df1['pf_SR'] # prices for fishers in Santa Rosalia
ys = df1['M_new'] # proportion of migrated squid
ml = df1['ML'] # mantle length
sst = df1['sst_anom'] # sea surface temperature anomaly
cg = df1['C_GY'] # catches from datamares in Guaymas
cr = df1['C_RM'] # catches from datamares in all other offices except SR, GY
cs = df1['C_SR'] # catches from datamares in Santa Rosalia
ca = df1['C_ALL'] # catches from datamares summed over all offices

df2 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_price_pfpe_ratio.xlsx', sheetname='Sheet1')
#! load columns
pm = df2['pm'] # comtrade price per ton
psr = df2['pf_sr'] # fishers price SR
pgy = df2['pf_gy'] # fishers price SR
pre = df2['pf_re'] # fishers price Remaining offices

### Plot files #################################################################
### catches
fig = plt.figure()
# add the first axes using subplot
ax1 = fig.add_subplot(111)
line1, = ax1.plot(cat, label = "Total", color="black")
line2, = ax1.plot(cg, label = "Guaymas", color = "blue")
line3, = ax1.plot(cs, label = "Santa Rosalia", color = "red")
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 12)
# ax1.set_xlim(0,)
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
line1, = ax1.plot(pg, label = "Guaymas", color = "blue")
line2, = ax1.plot(ps, label = "Santa Rosalia", color = "red")
line3, = ax1.plot(pr, label = "Other offices", color="black")
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 12)
ax1.set_xlim(0,len(yr))
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("Fishers' price $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
# legend
plt.legend([line3, line2, line1], ["Remaining offices", "Guaymas", "Santa Rosalia"], fontsize= 11, loc="best")
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_price.png',dpi=500)
plt.show()

### migration
fig = plt.figure()
# add the first axes using subplot
ax1 = fig.add_subplot(111)
line1, = ax1.plot(ys, color = "black")
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 12)
ax1.set_xlim(0,len(yr))
# y-axis
plt.ylabel("Proportion of migrated squid", rotation=90, labelpad=5, fontsize=20, **hfont)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_M.png',dpi=500)
plt.show()

### sst anomaly
fig = plt.figure()
# add the first axes using subplot
ax1 = fig.add_subplot(111)
line1, = ax1.plot(sst, color = "black")
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 12)
ax1.set_xlim(0,len(yr))
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("SST anomaly", rotation=90, labelpad=5, fontsize=20, **hfont)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_sst.png',dpi=500)
plt.show()

### prices and catch scatter
x = range(0,130000)
y = range(0,18000)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(cg,pg, s= 30, c='b', marker='o', label='Guaymas')
ax1.scatter(cs,ps, s= 30, c='y', marker='o', label='Santa Rosalia')
ax1.scatter(cr,pr, s= 30, c='r', marker='o', label='Remaining offices')
# axis
ax1.set_xlim(0,5E4)
plt.xlabel("Catch $tons$",fontsize=20)
plt.ylabel("Fishers' price $MXN$",fontsize=20)
plt.legend(loc="best", fontsize=12);
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_scatterprice.png',dpi=500)
plt.show()

### ML and catch scatter
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ml[5:24], cat[5:24])
print("r-squared ml catch:", r_value**2)
scipy.stats.pearsonr(ml[5:24], cat[5:24])
# begin fit
x = ml[5:24]
y = cat[5:24]
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)  # fit_fn is now a function which takes in x and returns an estimate for y
# plot function
plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.xlim(20, 80)
plt.xlabel("Mantle length $cm$",fontsize=20)
plt.ylim(0, 130E3)
plt.ylabel("Catch $tons$",fontsize=20)
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_scatterml.png',dpi=500)
plt.show()

### SST anomaly
fig = plt.figure()
# add the first axes using subplot populated with predictions
ax1 = fig.add_subplot(111)
line1, = ax1.plot(cat, label = "Catch", color = "sage", linewidth=3)
# add the second axes using subplot with sst
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line2, = ax2.plot(sst, color="grey", linewidth=3)
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 12)
ax2.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 12)
# y-axis
ax1.set_ylabel("Catch $tons$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax2.set_ylabel("SST anomaly $^\circ$C", rotation=270, color='grey', labelpad=22, fontsize=20, **hfont)
plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
# legend
plt.legend([line1, line2], ["Catch", "SST anomaly"], fontsize= 12)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_sstcatch.png',dpi=500)
plt.show()

### price gap
x = range(18000)
y = range(5000)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(pm, psr, s=30, c='red', marker="o", label='Santa Rosalia')
ax1.scatter(pm, pgy, s=30, c='blue', marker="o", label='Guaymas')
ax1.scatter(pm, pre, s=30, c='black', marker="o", label='Remaining offices')
# plt.xlim(20, 80)
plt.xlabel("Fishers' price $MXN$",fontsize=20)
# plt.ylim(0, 130E3)
plt.ylabel("Market price $MXN$",fontsize=20)
plt.legend(loc="best", fontsize=12);
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_pmpe.png',dpi=500)
plt.show()
