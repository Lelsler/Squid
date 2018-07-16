### Load data ##################################################################
###! Model outputs
R_C1 = np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelR_C_20180411.npy")
R_pf1 = np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelR_pf_20180411.npy")
meanNoR_C =np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanC.npy")
meanNoR_P =np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanP.npy")

NoR_C1 = np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelNoR_C_20180411.npy")
NoR_pf1 = np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/ModelNoR_pf_20180411.npy")
meanR_C =np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_R_meanC.npy")
meanR_P =np.load("/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_R_meanP.npy")

# Exclude first data point
R_C = R_C1[1:]
R_pf = R_pf1[1:]

NoR_C = NoR_C1[1:]
NoR_pf = NoR_pf1[1:]

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
ax1.scatter(meanR_C, meanR_P, s=30, c='b', marker="o", label='BEM+')
ax1.scatter(meanNoR_C, meanNoR_P, s=30, c='y', marker="o", label='BEM')
ax1.scatter(VolSR, PrSR, s=30, c='r', marker="s", label='SR data')
ax1.scatter(VolAll, PrAll, s=30, c='g', marker="s", label='All offices data')
# x-axis
plt.xlabel("Catch $tons$",fontsize=20, **hfont)
plt.xlim(1,1E5)
# y-axis
plt.ylabel("Price for fishers in MXN",fontsize=20, **hfont)
plt.ylim(1,)
# legend
plt.legend(loc="best", fontsize=10);
# save &show stuff
# fig.savefig('/Users/lauraelsler/Dropbox/PhD/Resources/P2/Squid/CODE/Squid/FIGS/R1_20180411.png',dpi=200)
plt.show()

###! Time series plot
fig = plt.figure()
a, = plt.plot(R_C, label = "R catch")
b, = plt.plot(NoR_C, label = "NoR catch")
e, = plt.plot(VolAll, label = "data catch")
c, = plt.plot(R_pf, label = "R pf")
d, = plt.plot(NoR_pf, label = "NoR pf")
f, = plt.plot(PrAll, label = "data price")
plt.xlim(2,len(R_C)-2)
#plt.ylim(0,3)
plt.xlabel("yr",fontsize=20)
plt.ylabel("variables",fontsize=20)
plt.legend(handles=[a,b,c,d,e,f], loc='best')
plt.title("Test", fontsize= 25)
plt.show()
