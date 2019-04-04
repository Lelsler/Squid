#### Laura Elsler: February 2019
#### Import packages ###########################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
import scipy.stats as stats
from pandas import *
hfont = {'fontname':'Helvetica'}

################################################################################
###########################  LOAD DATA   #######################################
################################################################################

# load data for data plots
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_all_plot_data.xlsx', sheetname='Sheet1')
yr = df1['year'] # year
cat = df1['C'] # catches total from Tim data 1991-2014
pg = df1['pf_GY'] # prices for fishers in Guaymas
pr = df1['pf_RM'] # prices for fishers in all other offices except SR, GY
ps = df1['pf_SR'] # prices for fishers in Santa Rosalia
ys = df1['M_new'] # proportion of migrated squid
ml = df1['ML'] # mantle length
# sst = df1['sst_anom'] # sea surface temperature anomaly
cg = df1['C_GY'] # catches from datamares in Guaymas
cr = df1['C_RM'] # catches from datamares in all other offices except SR, GY
cs = df1['C_SR'] # catches from datamares in Santa Rosalia
ca = df1['C_ALL'] # catches from datamares summed over all offices

# load data for price plots
df2 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/squid_price_pfpe_ratio.xlsx', sheetname='Sheet1')
pm = df2['pm'] # comtrade price per ton
psr = df2['pf_sr'] # fishers price SR
pgy = df2['pf_gy'] # fishers price SR
pre = df2['pf_re'] # fishers price Remaining offices

# load data for optimization plots
df4 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/model_compare.xlsx', sheetname='Sheet1')
bemc = df4['CBEM'] # BEM catch predictions
edmc = df4['CEDM'] # EDM catch predictions
mlmc = df4['CMLM'] # MLM catch predictions
datc = df4['CData'] # data
bemp = df4['PfBEM'] # BEM price predictions
edmp = df4['PfEDM'] # EDM price predictions
mlmp = df4['PfMLM'] # MLM price predictions
datp = df4['PfData'] # data

# load data for SST anomaly plots
df5 = pd.read_csv('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/SST_anomaly.csv', usecols=['anom','esst_avg','time'])
yr1 = df5['time'] # BEM catch predictions
sst = df5['anom'] # BEM catch predictions
ssta = df5['esst_avg'] # BEM catch predictions

################################################################################
###########################  PLOT FILE   #######################################
################################################################################

### prepare dataframe for plot SST and catch
df5.time = df5.time.round()
df6 = df5.drop_duplicates(subset=['time', 'esst_avg'], keep='first')
df6 = df6[9:]
df6 = df6.reset_index()
sstx = df6['esst_avg']
cat1= cat[:-1]

############## MAIN TEXT MAP ###################################################
#! SST anomaly
fig = plt.figure()
# add the first axes using subplot populated with predictions
ax1 = fig.add_subplot(111)
line1, = ax1.plot(cat1, label = "Catch", color = "sage", linewidth=3)
# add the second axes using subplot with sst
ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
line2, = ax2.plot(sstx, color="grey", linewidth=3)
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 12)
# ax1.set_xlim(0,len(cat1))
ax2.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 12)
# ax2.set_xlim(0,len(sstx))
# y-axis
ax1.set_ylabel("Catch $tons$", rotation=90, labelpad=5, fontsize=22, **hfont)
ax1.set_yticklabels(np.arange(0,140000,20000), rotation=45, fontsize= 12)
ax2.set_ylabel("SST anomalies $^\circ$C", rotation=270, color='grey', labelpad=22, fontsize=20, **hfont)
ax2.set_ylim(-2,1.5)
ax2.set_yticklabels(np.arange(-2,1.5,0.5), fontsize= 12)
plt.gcf().subplots_adjust(bottom=0.15,right=0.9)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
# legend
# plt.legend([line1, line2], ["Catch", "SST anomaly"], fontsize= 12)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_sstcatch.png',dpi=500)
plt.show()


######## SI OPTIMIZATION #######################################################
#! optimization catch
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
# add the first axes using subplot
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(bemc, label = "BEM", color="steelblue")
line2, = ax1.plot(edmc, label = "EDM", color="sage", linewidth=3)
line3, = ax1.plot(mlmc, label = "MLM", color="orange")
line4, = ax1.plot(datc, label = "Data", color="indianred", linewidth =2)
# x-axis
ax1.set_xticklabels(np.arange(1996,2017,5), rotation=45, fontsize= 15)
plt.yticks(fontsize= 15, rotation=45)
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("Catch $tons$", rotation=90, labelpad=5, fontsize=20, **hfont)
# legend
plt.legend([line1, line2,line3,line4], ["BEM","EDM","MLM","Data"], fontsize= 12)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_optic.png',dpi=500)
plt.show()

#! optimization fishers price
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
# add the first axes using subplot
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(bemp, label = "BEM", color="steelblue", linewidth=2)
line2, = ax1.plot(edmp, label = "EDM", color="sage", linewidth=2)
line3, = ax1.plot(mlmp, label = "MLM", color="orange", linewidth=2)
line4, = ax1.plot(datp, label = "Data", color="indianred", linewidth=2)
line5, = ax1.plot(datc+5E5, label = "Data", color="indianred", linewidth=2)
# x-axis
ax1.set_xticklabels(np.arange(1996,2017,5), rotation=45, fontsize= 15)# ax1.set_xlim(0,)
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("Fishers' price $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
ax1.set_ylim(0,3.5E4)
plt.yticks(fontsize= 15, rotation=45)
# legend
plt.legend([line1, line2,line3,line4], ["BEM","EDM","MLM","Data"], loc='best', fontsize= 12)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_optipf.png',dpi=500)
plt.show()

############## SI STATES #######################################################
#! catches
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
# add the first axes using subplot
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(cat, label = "Total", color="black", linewidth=2)
line2, = ax1.plot(cg, label = "Guaymas", color = "blue", linewidth=2)
line3, = ax1.plot(cs, label = "Santa Rosalia", color = "red", linewidth=2)
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 15)
# ax1.set_xlim(0,)
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("Catches $tons$", rotation=90, labelpad=5, fontsize=20, **hfont)
plt.yticks(fontsize= 15, rotation=45)
# legend
plt.legend([line1, line2, line3], ["Total", "Guaymas", "Santa Rosalia"], fontsize= 11)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_catch.png',dpi=500)
plt.show()

### fishers price
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
# add the first axes using subplot
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(pg, label = "Guaymas", color = "blue", linewidth=2)
line2, = ax1.plot(ps, label = "Santa Rosalia", color = "red", linewidth=2)
line3, = ax1.plot(pr, label = "Other offices", color="black", linewidth=2)
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 15)
ax1.set_xlim(0,len(yr))
# plt.xlabel("Year",fontsize=20, **hfont)
# y-axis
plt.ylabel("Fishers' price $MXN$", rotation=90, labelpad=5, fontsize=20, **hfont)
plt.yticks(fontsize= 15, rotation=45)
# legend
plt.legend([line3, line2, line1], ["Remaining offices", "Guaymas", "Santa Rosalia"], fontsize= 11, loc="best")
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_price.png',dpi=500)
plt.show()

### mantle length
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
# add the first axes using subplot
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(ml[5:], color = "black", linewidth=2)
# x-axis
ax1.set_xticklabels(np.arange(1991,2017,5), rotation=45, fontsize= 15)
ax1.set_xlim(0,len(yr))
# y-axis
plt.ylabel("Mantle length $cm$", rotation=90, labelpad=5, fontsize=20, **hfont)
plt.yticks(fontsize= 15, rotation=45)
# save and show
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_ML.png',dpi=500)
plt.show()

### sst anomaly
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
# add the first axes using subplot
ax1 = fig1.add_subplot(111)
ax2 = fig1.add_subplot(111)
ax1.scatter(yr1,sst, s= 30, c='black', marker='.', label='Measurements', linewidth=2)
ax2.plot(yr1,ssta, c='red', linestyle='-', label='Yearly average', linewidth=2)
# axis
ax1.set_xticklabels(np.arange(1975,2025,5), rotation=45, fontsize= 15)
ax2.set_xticklabels(np.arange(1975,2025,5), rotation=45, fontsize= 15)
# ax1.set_xlim(0,len(yr1)+12)
plt.ylabel("SST anomalies $^\circ$C", rotation=90, labelpad=5, fontsize=20, **hfont)
plt.yticks(fontsize= 15, rotation=45)
plt.legend(loc="best", fontsize=15)
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_sst.png',dpi=500)
plt.show()

############## SI PRICES #######################################################
#! prices and catch scatter
x = range(0,130000)
y = range(0,18000)
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.15,left=0.15,right=0.9)
# add the first axes using subplot
ax1 = fig1.add_subplot(111)
ax1.scatter(cs,ps, s= 30, c='red', marker='o', label='Santa Rosalia')
ax1.scatter(cg,pg, s= 30, c='blue', marker='o', label='Guaymas')
ax1.scatter(cr,pr, s= 30, c='black', marker='o', label='Remaining offices')
# axis
ax1.set_xlim(0,5E4)
plt.xlabel("Catch $tons$",fontsize=20)
plt.xticks(fontsize= 15)
plt.ylabel("Fishers' price $MXN$",fontsize=20)
plt.yticks(fontsize= 15, rotation=45)
plt.legend(loc="best", fontsize=12);
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_scatterprice.png',dpi=500)
plt.show()

#! price gap
x = range(18000)
y = range(5000)
fig1 = plt.figure(figsize=[9,6])
gs = gridspec.GridSpec(1,1,bottom=0.2,left=0.15,right=0.9)
# add the first axes using subplot
ax1 = fig1.add_subplot(111)
ax1.scatter(pm, psr, s=30, c='red', marker="o", label='Santa Rosalia')
ax1.scatter(pm, pgy, s=30, c='blue', marker="o", label='Guaymas')
ax1.scatter(pm, pre, s=30, c='black', marker="o", label='Remaining offices')
# plt.xlim(20, 80)
plt.xlabel("Export price $MXN$",fontsize=20)
plt.xticks(fontsize= 15)
# plt.ylim(0, 130E3)
plt.ylabel("Fishers' price $MXN$",fontsize=20)
plt.yticks(fontsize= 15, rotation=45)
plt.legend(loc="best", fontsize=12);
# fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/SI_pmpe.png',dpi=500)
plt.show()




################### NOT USED ###################################################

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
