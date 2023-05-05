using EpiCont

using Random, Distributions, LinearAlgebra
using Base.Threads
using PyCall, Conda

plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")
sns = pyimport("seaborn")

bayes = pyimport("bayes_opt")
bys = bayes.bayesian_optimization

#Optimisation parameters and bounds
pbounds = py"{'pred_days': (7, 35)}"

#we pass th eparameters as one structure between functions

mutable struct parameters
    R0
    δ
    I0
    ndays
    ρ
    ρvar
    repd_mean
    nYdel
    N
    ctrl_states
    R_coeff
    cost_sel
    use_inc
    cost_of_state
    Lc_target
    Lc_target_pen
    R_target
    alpha
    beta
    ovp
    γ
    rf
    R_est_wind
    use_S
    pred_days
    days
    dec
    policies_tpl
    policies
    I_min
    n_ens
    delay_calc_v 
    binn
    distr_sel
    under_rep_calc
end

#Define the parameters of the epidemic
#Disease A (Covid-like)

R0 = 3.5 #Basic Reproduction number
gen_time = 6 #Generation time (in days)
δ = 0.08 #Death rate
I0 = 10 #initial no. of infections
ndays = 21*7 #epidemic length
ρ, ρvar = 0.6, 0. #Under reporting, mean/variance
dρ = Normal(ρ, ρvar)
repd_mean, repd_var = 6, 2 #Reporting delay, mean/variance

# #Define the parameters of the epidemic
#Disease B (Ebola-like)

# R0 = 2.5 #Basic Reproduction number
# gen_time = 11 #Generation time (in days)
# δ = 0.08 #Death rate
# I0 = 10 #initial no. of infections
# ndays = 31*7 #epidemic length
# ρ, ρvar = 1.0, 0. #Under reporting, mean/variance
# dρ = Normal(ρ, ρvar)
# repd_mean, repd_var = 12, 2 #Reporting delay, mean/variance

N = 1e6 #Total population

# Setting-up the control (using non-pharmaceutical interventions)
ctrl_states = ["No restrictions", "Social distancing", "Lockdown"]
R_coeff = [1.0, 0.5, 0.2] #R0_act = R0 * ctrl_states
I_min = 100 #minimum treshold for action

#Sim and control options
cost_sel = 1 #1: bilinear+oveshoot, 2: flat+quadratic for overshoot
use_inc = 1 #1: control for incidence, 0: control for infectiousness
delay_calc_v = 0 #1: as in ref, 0: from incidence
under_rep_calc = 0 #1: as in ref, 0: separately, from incidence
distr_sel = 0 #0: Deterministic, 1: Poisson, 2: Binomial
delay_on = 0 #1: sim with time-delay, 0: no-delay (if 0, set delay_calc_v = 0)

binn = 10 
#est_R = 1 #1: when making predictions use R estimated from data. 0: use model R

#The cost function
cost_of_state = [0.0, 0.01, 0.15] #Cost of interventions/day
Lc_target = 5000 #desired infectiousness
Lc_target_pen = Lc_target*1.5 #extra overshoot penalty
R_target = 1.0
alpha = 1.587/Lc_target #~proportional gain (regulates error)
beta = 0.0 #~derivative gain (regulates error velocity)
cost_of_state = [0.0, 0.01, 0.15]
ovp = 5.0 #overshoot penalty
γ = 0.95 #discounting factor

#Simulation parameters
n_ens = 1 #MC assembly size for prediction
sim_ens = 100 #assembly size for full simulation

#Frequecy of policy review
rf = 7 #days
R_est_wind = rf-2 #window for R estimation
use_S = 0

#Prediction window
pred_days = rf*3
pred_days_max = rf*5
days = 1:ndays+pred_days_max

#Distribution of the reporting delay

Ydel = Erlang(repd_mean, 1)
Ydelpdf = pdf.(Ydel,days)
cdelpdf = sum(Ydelpdf)
nYdel = Ydelpdf/cdelpdf

if delay_on == 0
    nYdel = zeros(length(days))
    nYdel[1] = 1.0
end


#decisions considered
dec = 1

nstates, policies, policies_tpl = init_policies(ctrl_states,dec)

par = parameters(R0,
δ,
I0,
ndays,
ρ,
ρvar,
repd_mean,
nYdel,
N,
ctrl_states,
R_coeff,
cost_sel,
use_inc,
cost_of_state,
Lc_target,
Lc_target_pen,
R_target,
alpha,
beta,
ovp,
γ,
rf,
R_est_wind,
use_S,
pred_days,
days,
dec,
policies_tpl,
policies,
I_min,
n_ens,
delay_calc_v,
binn,
distr_sel,
under_rep_calc)

reward = reward_sel(1)

Y = Erlang(Int(round(gen_time/2.0)), 2.0)

Ypdf = pdf.(Y,days)
cpdf = sum(Ypdf)
nY = Ypdf/cpdf

#Risk of death wrt. time of becoming infectious

Yd = Erlang(7, 3.0)

Ydpdf = pdf.(Yd,days)
cdpdf = sum(Ydpdf)
nYd = Ydpdf/cdpdf

##

alpha_c = 1.3 #~proportional gain (regulates error)
beta_c = 0.0 #~derivative gain (regulates error velocity)

function Φ(Lcvect, Revect, policy)

    over_pen = zeros(length(Lcvect))
    cfs = zeros(length(Lcvect))
    
    for ii in 1:(length(Lcvect))
        cfs[ii] = cost_of_state[policy[ii]]
        if Lcvect[ii] > Lc_target*1.5
            over_pen[ii] = ovp
        end
    end

    res = -alpha_c./Lc_target * norm(Lcvect[1:ndays] .- Lc_target,1) -beta_c * norm(Revect[1:ndays] .- 1.0,1) - norm(cfs[1:ndays], 1) - norm(over_pen[1:ndays], 1) 
end

function epi_opt(;pred_days)

    pred_ds = Int(round(pred_days))
    dec = Int(round(floor(pred_ds/rf)))

    nstates, policies, policies_tpl = init_policies(ctrl_states,dec)

    rf_ds = rf

    par0 = par
    par0.pred_days = pred_ds
    par0.rf = rf_ds
    par0.R_est_wind = rf_ds-2
    par0.dec = dec
    par0.policies = policies
    par0.policies_tpl = policies_tpl

    #Initialise
    Ivect = zeros(ndays+pred_ds, sim_ens)
    Svect = zeros(ndays+pred_ds, sim_ens) #Susceptibles
    Revect = zeros(ndays+pred_ds, sim_ens) #Effective Reproduction numer
    Rewvect = zeros(ndays+pred_ds, sim_ens) #Effective Reproduction numer -- weighted
    Lvect = zeros(ndays+pred_ds, sim_ens) #Total infectiousness
    Lcvect = zeros(ndays+pred_ds, sim_ens) #Total infectiousness with perceived infection numbers -- control-basis
    cvect = zeros(ndays+pred_ds, sim_ens) #Reported cases
    Dvect = zeros(ndays+pred_ds, sim_ens) #Deceased
    Ldvect = zeros(ndays+pred_ds, sim_ens)
    Rest = zeros(ndays+pred_ds, sim_ens) #estimated Re from data 
    R0est = zeros(ndays+pred_ds, sim_ens) #estimated R0 from data 

    policy = ones(Int8, ndays+pred_ds, sim_ens)

    #Initialisation
    Ivect[1,:] .= I0
    Lvect[1,:] .= I0
    Dvect[1,:] .= 0
    cvect[1,:] .= trunc(Int, (Ivect[1]*rand(dρ, 1)[1]))
    Lcvect[1,:] .= cvect[1]
    Svect[1,:] .= N-I0
    Revect[1,:] .= R0
    Rewvect[1,:] .= R0
    Rest[1,:] .= 1.0
    R0est[1,:] .= 1.0

    reward_ens = zeros(sim_ens)

    Threads.@threads for ii in 1:sim_ens


        Ivect[:,ii], Revect[:,ii], Lvect[:,ii], Lcvect[:,ii], Dvect[:,ii], Svect[:,ii], cvect[:,ii], Rewvect[:,ii], policy[:,ii], Rest[:,ii], R0est[:,ii] = EpiRun_preds(Ivect[:,ii], Revect[:,ii], Lvect[:,ii], Lcvect[:,ii], Ldvect[:,ii], Dvect[:,ii], Svect[:,ii], cvect[:,ii], Rewvect[:,ii], policy[:,ii], Rest[:,ii], R0est[:,ii], nY, nYd, reward, par0)
        
        reward_ens[ii] = Φ(Lcvect[:,ii], Rewvect[:,ii], policy[:,ii])
    end
    return mean(reward_ens)
    
end

epi_opt(;pred_days=18.0)

## optimisation

optimizer = bys.BayesianOptimization(
    f=epi_opt,
    pbounds=pbounds,
    random_state=2,
)

res = optimizer.maximize(
    init_points=8,
    n_iter=20,
)

optimizer.max

##

#plotting the iterates
alphas = zeros(length(optimizer.res))
targets = zeros(length(optimizer.res))

for ii in 1:length(alphas)
    targets[ii] = get(optimizer.res[ii], "target", 0.0)
    alphas[ii] = get(get(optimizer.res[ii], "params", 0.0), "pred_days", 0.0)
end

plt = pyimport("matplotlib.pyplot")

fig = plt.figure()
#ax = fig.add_subplot(projection="3d")
plt.scatter(alphas, targets)
#plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
plt.xlabel("pred_days")
plt.ylabel("objective function")
plt.show()

fig = plt.figure()
#ax = fig.add_subplot(projection="3d")

np = pyimport("numpy")

X = np.arange(7, 35, 0.05)

obs = []
for ii in 1:length(targets)
    append!(obs, [[alphas[ii]]])
end

pts = []
for ii in 1:length(X)
    append!(pts, [[X[ii]]])
end

optimizer._gp.fit(obs, targets)
fit, sdv = optimizer._gp.predict(pts, return_std=true)

fit

# Plot the surface.
plt.scatter(alphas, targets)
plt.plot(X,fit)
#plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
plt.xlabel("pred_days")
plt.ylabel("objective function")

plt.show()