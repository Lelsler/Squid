### Load data ##################################################################
###! Model outputs
R_C1 = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/Old/PY/DATA/ModelR_C_20180411.npy")
R_pf1 = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/Old/PY/DATA/ModelR_pf_20180411.npy")

NoR_C1 = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/Old/PY/DATA/ModelNoR_C_20180411.npy")
NoR_pf1 = np.load("./Dropbox/PhD/Resources/P2/Squid/CODE/Old/PY/DATA/ModelNoR_pf_20180411.npy")

# Exclude first data point
R_C = R_C1[1:]
R_pf = R_pf1[1:]

NoR_C = NoR_C1[1:]
NoR_pf = NoR_pf1[1:]

###! Load data
df1 = pd.read_excel('./Dropbox/PhD/Resources/P2/Squid/Laura/PriceVolDataCorrected.xlsx', sheetname='Sheet1')

# Load columns
VolAll = df1['tons_DM']
VolEtal = df1['tons_DM_etal']
VolSR = df1['tons_DM_SR']

PrAll = df1['priceMXNia_DM']
PrEtal = df1['priceMXNia_DM_etal']
PrSR = df1['priceMXNia_DM_SR']

#### PLOT ######################################################################
###! Scatter plot
x = range(100000)
y = range(0,4)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(R_C, R_pf, s=30, c='b', marker="s", label='model with Rtt')
ax1.scatter(NoR_C, NoR_pf, s=30, c='y', marker="o", label='model w/o Rtt')
ax1.scatter(VolAll, PrEtal, s=30, c='g', marker="o", label='Others data')
ax1.scatter(VolAll, PrSR, s=30, c='r', marker="s", label='SR data')
plt.title("Price/catch: model and data", fontsize= 25)
plt.xlabel("Catch in t",fontsize=20)
plt.ylabel("Price for fishers in MXN",fontsize=20)
plt.legend(loc="best", fontsize=10);
# fig.savefig('./Dropbox/PhD/Resources/P2/Squid/CODE/Old/PY/FIGS/R1_20180411.png',dpi=200)
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
#fig.savefig('./Dropbox/PhD/Deliverables/3_March/Week1/CpfPred_Rtt.png',dpi=200)
plt.show()
