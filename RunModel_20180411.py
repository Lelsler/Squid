
##### Initiate arrays ##########################################################
sim = np.arange(0,100) # number of simulations
x = np.zeros(12) # set array to save parameters
par = np.zeros((sim.shape[0],x.shape[0])) # matrix to save parameter values of each simulation
cat = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save catches in each time period of each simulation
pri = np.zeros((sim.shape[0],tau.shape[0])) # matrix to save prices in each time period of each simulation

##### Run the model ############################################################
for j in range(0,sim.shape[0]): # draw randomly a float in the range of values for each parameter
    # a1 = np.random.uniform(2E-08, 3E-08) # parameter ranges
    # a2 = np.random.uniform(0.2, 9E-16)
    b1 = np.random.uniform(38.750, 42.1)
    b2 = np.random.uniform(-3.987, -6.9)
    b3 = np.random.uniform(11.478, 16.4)
    beta = np.random.uniform(0.01, 0.1)
    c_p = np.random.uniform(1000, 2148)
    g = np.random.uniform(0, 2.9)
    gamma = np.random.uniform(20000, 51000)
    # l1 = np.random.uniform(-0.0005, -0.0122)
    # l2 = np.random.uniform(0.0317, 0.7927)
    m = np.random.uniform(2368793, 8450159)
    w_m = np.random.uniform(11956952, 28108539)

    x = [a1, b1, b2, b3, beta, c_p, g, gamma, l1, l2, m , w_m]
    par[j] = x

    OUT = np.zeros(tau.shape[0])
    OUT1 = np.zeros(tau.shape[0])

    for i in np.arange(1,tmax):
            tau, ML, q, y_S, R_tt, S, c_t, E, C, p_e, p_f, R = model(b1, b2, b3, n1, n2, l1, l2, a1, g, K, m, f, B_h, B_f, h1, h2, gamma, beta, c_p, w_m, flag)
            OUT[i]= p_f[i]
            OUT1[i]= C[i]
            pri[j,i] = OUT[i]
            cat[j,i] = OUT1[i]


lowC = np.zeros(y.shape[0]) # initiate variables for 95% confidence interval
highC = np.zeros(y.shape[0])
meanC = np.zeros(y.shape[0])
lowP = np.zeros(y.shape[0])
highP = np.zeros(y.shape[0])
meanP = np.zeros(y.shape[0])

for h in range(0,y.shape[0]): # calculate the 95% confidence interval
    z = cat[:,h]
    lowC[h] = np.nanmean(z) - ((1.96 * np.nanstd(z))/np.sqrt(np.count_nonzero(~np.isnan(z))))
    highC[h] = np.nanmean(z) + ((1.96 * np.nanstd(z))/np.sqrt(np.count_nonzero(~np.isnan(z))))
    meanC[h] = np.nanmean(z)
    zeta = pri[:,h]
    lowP[h] = np.nanmean(zeta) - ((1.96 * np.nanstd(zeta))/np.sqrt(np.count_nonzero(~np.isnan(zeta))))
    highP[h] = np.nanmean(zeta) + ((1.96 * np.nanstd(zeta))/np.sqrt(np.count_nonzero(~np.isnan(zeta))))
    meanP[h] = np.nanmean(zeta)

##### Save data  ###############################################################
if flag == 0:
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support2_95_NoR_lowC.npy", lowC)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support2_95_NoR_highC.npy", highC)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_NoR_lowP.npy", lowP)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_NoR_highP.npy", highP)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanP.npy", meanP)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_NoR_meanC.npy", meanC)
if flag == 1:
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support2_95_R_lowC.npy", lowC)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support2_95_R_highC.npy", highC)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_R_lowP.npy", lowP)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_R_highP.npy", highP)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_R_meanP.npy", meanP)
    np.save("./Dropbox/PhD/Resources/P2/Squid/CODE/Squid/DATA/R1support1_95_R_meanC.npy", meanC)
