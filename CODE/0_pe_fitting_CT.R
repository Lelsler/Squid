##################################################################
## Read in data ##################################################
##################################################################
data <- read.csv('/Users/lauraelsler/Dropbox/PhD/Resources/Squid/documents/data_interpretation/WAvg_yr_EACT.csv',header=TRUE) #read in EXPORT VOLUME AND VALUE data
pe<-data.frame(data) # put data in data frame pe
pe[pe == 1.] <- NA # make all 1s -> NAs

# Year,CT_Tons,CT_AvgMXNt,CT_WavgMXNt,EA_Tons,EA_AvgMXNiat,EA_WavgMXNiat

k <- na.omit(pe) # omit NAs, put in k 
plot(k$CT_Tons,k$CT_AvgMXNt) # plot value over volume, EA
plot(k$CT_Tons,k$CT_WavgMXNt) # plot value over volume, CT

## Avg
fit <- nls(CT_AvgMXNt ~ gamma*I(CT_Tons^(-1*b)), data=k, start = list(gamma=1000,b=-0.25), trace = T) # 

s <- seq(0,max(k$CT_Tons), length=1000)

lines(s, predict(fit, list(CT_Tons=s)), col="blue", lty =2)

## Wavg
fit2 <- nls(CT_WavgMXNt ~ gamma*I(CT_Tons^(-1*b)), data=k, start = list(gamma=1000,b=0.25), trace = T) # 

s <- seq(0,max(k$CT_Tons), length=1000)

lines(s, predict(fit2, list(CT_Tons=s)), col="red", lty =2)
