import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from pylab import *

###### LOAD DATA ###############################################################
df1 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/R3_data.xlsx', sheetname='Sheet1')

# load columns
yr = df1['year']
pe = df1['pe_MXNiat']
pf = df1['pf_MXNiat']
C = df1['C_t']
T = df1['essh_avg']
p = pf/pe

###### LOAD DATA ###############################################################
df2 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/DATA/composite_outside_catches.xlsx', sheetname='composite_outside_catches')

# load columns
yr2 = df2['year']
C_out = df2['outside_catch']

###### LOAD DATA ###############################################################
df3 = pd.read_excel('./Dropbox/PhD/Resources/Squid/Squid/Laura/SST_anomaly.xlsx', sheetname='Sheet1')

# load columns
yr3 = df3['year']
sst = df3['esst_avg']

# sts = np.array(sst, dtype=pd.Series)
# tss = np.zeros(25)
# sts[9:]= tss
#
# copyto(sts[9:],tss)
# sst[9:] = np.empty_like (b)
# a[:] = b


###### PLOTS ###################################################################
###! Time series plot
hfont = {'fontname':'Helvetica'}

# create the general figure
fig1 = figure()
# and the first axes using subplot populated with data
ax1 = fig1.add_subplot(111)
line1, = ax1.plot(sst, color="orange")
ylabel("SST anomaly $m$", fontsize=20, **hfont)
xlabel("year", fontsize=20, **hfont)
ax1.set_xticklabels(np.arange(1991,2017,5))
ax1.set_xlim(right=2016)
# now, the second axes that shares the x-axis with the ax1
ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
line2, = ax2.plot(C, color="steelblue")
line3, = ax2.plot(C_out,color="indianred")
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
# y-axis 2
ax2.set_ylabel("Catches $tons$", rotation=270, color='k', labelpad=15, fontsize=20, **hfont)
# for the legend, remember that we used two different axes so, we need to build the legend manually
plt.legend([line1, line2, line3], ["SST anomaly", "Total catches", "Catches outside Gulf"], fontsize= 11)
savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R3_support2.png',dpi=500)
show()

###### STATISTICS ##############################################################
# T = T[2:26]
# C = C[2:26]
# C_out = C_out[2:]
#
# from scipy.stats import linregress
# linregress(T, C)
# linregress(T, C_out)

# # to get the coefficient of determination "r-squared:" r_value**2
# # r-squared (T,C): -0.18584996957060027
#
# rvalue=0.028448939264492042**2

###### 3D PLOT #################################################################
# ###! Plot 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# colmap = cm.ScalarMappable(cmap=cm.hsv)
# colmap.set_array(yr)
# p = ax.scatter(p,C,T,c=cm.hsv(yr), marker='o')
# cb = fig.colorbar(colmap)
# ax.set_xlabel('relative price pf/pe$')
# ax.set_ylabel('catch')
# ax.set_zlabel('sea surface height')
# # fig.savefig('./Dropbox/PhD/Resources/Squid/Squid/CODE/Squid/FIGS/R3_20180411.png',dpi=500)
# plt.show()
