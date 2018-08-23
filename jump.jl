using JuMP
using Ipopt
using DataFrames
import XLSX

abstract type Model end
struct Direct <: Model end
struct Migration <: Model end

#run = Direct();
run = Migration();

#Initial isotherm coefficient guesses
b10 = 41.75; #mean depth
b20 = -5.696; #amplitude
b30 = 16.379; #period of seasonal cycles
B_h = 7.203 # hours per fisher
B_f = 2.0 # fisher per panga
f = 40.0 # l of fuel per trip
m = 5_492_603.58 # cost per unit of transport all boats, MXN/trip
c_t = m*f # fleet cost of transport

#df2 looks at 2001-2016, so df1 should only yield that set too
df1 = DataFrame(XLSX.readtable("./DATA/R3_data.xlsx", "Sheet1")...)
# load columns
y = Int64.(df1[:year][11:26]) #
#pe = Float64.(map(x->ismissing(x) ? NaN : x, df1[:pe_MXNiat][11:26])) #
pf = Float64.(df1[:pf_MXNiat][11:26]) #
#ct = Float64.(map(x->ismissing(x) ? NaN : x, df1[:C_t][11:26])) #
#ssh = Float64.(map(x->ismissing(x) ? NaN : x, df1[:essh_avg][11:26])) #
ml = Float64.(df1[:ML][11:26]) #
ys = map(x->parse(Float64,x),df1[:y_S][11:26]) #


df2 = dropmissing(DataFrame(XLSX.readtable("./DATA/PriceVolDataCorrected.xlsx", "Sheet1")...))
VolAll = Float64.(df2[:tons_DM]) ## CATCH DATA
PrAll = Float64.(df2[:priceMXNia_DM]) ## PRICE DATA

### New max time
tmax = length(df2[:Year]);
(Cmax, CmaxIdx) = findmax(VolAll);

model = JuMP.Model(solver = IpoptSolver(max_iter=100000,print_frequency_iter=200,sb="yes"));


tauh(a,b,c,t) = a+b*cos.(t)+c*sin.(t);
ddtauh(b,c,t) = -b*cos.(t)-c*sin.(t);
function getMinTau(a,b,c)
    t = atan(c/b);
    if ddtauh(b,c,t) > 0
        tauh(a,b,c,t)
    else
        tauh(a,b,c,t+π)
    end
end
function getMaxTau(a,b,c)
    t = atan(c/b);
    if ddtauh(b,c,t) < 0
        tauh(a,b,c,t)
    else
        tauh(a,b,c,t+π)
    end
end
function getK(a,b,c,Cmax,idx)
    taul = getMinTau(a,b,c);
    tauu = getMaxTau(a,b,c);
    Cmax*0.1*((tauh(a,b,c,idx)-taul)/(tauu-taul))
end

JuMP.register(model, :tauh, 4, tauh, autodiff=true)
JuMP.register(model, :ddtauh, 3, ddtauh, autodiff=true)
JuMP.register(model, :getMinTau, 3, getMinTau, autodiff=true)
JuMP.register(model, :getMaxTau, 3, getMaxTau, autodiff=true)
JuMP.register(model, :getK, 5, getK, autodiff=true)

tauLower = getMinTau(b10,b20,b30);
tauUpper = getMaxTau(b10,b20,b30);
if tauLower < 20.0 || tauUpper > 80.0
    error("tau limits out of bounds 20.0 <= tau <= 80.0. Change b(1,2,3)0 variables");
end
K0 = getK(b10,b20,b30,Cmax,CmaxIdx);


@variable(model, 38.75 <= b1 <= 42.1, start = b10) # isotherm depth depth (est)
@variable(model, b2, start = b20) # isotherm amplitude (est)
@variable(model, b3, start = b30) # isotherm seasonal period (est)
@variable(model, 0.0 <= beta <= 1.0) # slope of demand-price function
@variable(model, 1000.0 <= c_p <= 2148.0) # cost of processing, MXNia/t
@variable(model, 0.0 <= g <= 3.2) # population growth rate
@variable(model, 20_000.0 <= gamma <= 51_000.0) # maximum demand, t
#@variable(model, h1) # E scale
#@variable(model, h2) # E scale

#@variable(model, a1, start=1/exp.(tauUpper-b10)); # proportion of migrating squid, 1/max(e^(tau-b1))
@variable(model, K, start=K0) # Carrying capacity in t. Cmax * q

@variable(model, 20.0 <= tau[t=1:tmax] <= 80.0) # temperature
@variable(model, maxTau, start=tauUpper) # q scale
@variable(model, minTau, start=tauLower) # q scale

@variable(model, 4_000.0 <= p_e[t=1:tmax] <= 100_000.0) # export price
@variable(model, 0.0 <= q[t=1:tmax] <= 0.1) # catchability squid population
@variable(model, 0.0 <= y_S[t=1:tmax] <= 1.0) # Proportion of squid migration from initial fishing grounds
#@variable(model, Escal[t=1:tmax]) # fishing effort
@variable(model, 0.0 <= E[t=1:tmax] <= 1.0) # fishing effort
@variable(model, S[t=1:tmax] >= 0.0) # size of the squid population 18000 , 1200000
@variable(model, C[t=1:tmax]  >= 0.0) # squid catch 10% of S
@variable(model, p_f[t=1:tmax]) # price for fishers
@variable(model, match)

if typeof(run) == Migration
    @variable(model, 11_956_952.0 <= w_m <= 28_108_539.0) # min wage per hour all fleet
    @variable(model, 0.0 <= R_tt[t=1:tmax] <= 1.0) # trader cooperation
    @variable(model, 5.0 <= p_min[t=1:tmax] <= 10_000.0) #MXN/t O(1300) is good.
end


@NLconstraint(model, maxTau == getMaxTau(b1,b2,b3)); # q scale
@NLconstraint(model, minTau == getMinTau(b1,b2,b3)); # q scale
#@NLconstraint(model, a1 == 1/exp(maxTau-b1));
@NLconstraint(model, K == getK(b1,b2,b3,Cmax,CmaxIdx));
@constraint(model, [t=1:tmax], tau[t] == tauh(b1,b2,b3,t));
@NLconstraint(model, [t=1:tmax], p_e[t] == gamma*(C[t])^(-beta));
#@NLconstraint(model, [t=1:tmax-1], Escal[t+1] == E[t]+p_f[t]*C[t]-c_t*(E[t]/(B_h+B_f)));
#@constraint(model, [t=1:tmax], E[t] == h1*Escal[t]+h2);
@NLconstraint(model, [t=1:tmax-1], S[t+1] == S[t]+g*S[t]*(1-(S[t]/K))-C[t]);
@NLconstraint(model, [t=1:tmax], C[t] == q[t]*E[t]*S[t]);
#@NLconstraint(model, [t=1:tmax], y_S[t] == a1*exp(tau[t]-b1));
@NLconstraint(model, [t=1:tmax], y_S[t] == exp(tau[t]-maxTau));
@NLconstraint(model, [t=1:tmax], q[t] == 0.1*((tau[t]-minTau)/(maxTau-minTau)));
if typeof(run) == Direct
    #First Model
    @constraint(model, [t=1:tmax], p_f[t] == p_e[t]-c_p);
else
    #Second model
    @NLconstraint(model, [t=1:tmax], p_min[t] == (E[t]*w_m)/C[t]);
    @constraint(model, [t=1:tmax], R_tt[t] == 1-y_S[t]);
    @constraint(model, [t=1:tmax], p_f[t] == (p_e[t]-c_p)*(1-R_tt[t])+R_tt[t]*p_min[t]);
end

#Minimise the Least Squares differences between our model and the data
@NLconstraint(model, match == sum(abs(p_f[t] - PrAll[t])^2+abs(C[t] - VolAll[t])^2 for t in 1:tmax));
@objective(model, Min, match);


status = solve(model);


# Add ML, R
