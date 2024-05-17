using EpiCont

using Random, Distributions
using Base.Threads
using PyCall, Conda

Random.seed!(1)

#Conda.add("seaborn")
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")
sns = pyimport("seaborn") 

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
    under_rep_on
    delay_on
    rho_beta_a
    rho_beta_b
end

#Define the parameters of the epidemic
#Disease A (Covid-like)

# R0 = 3.5 #Basic Reproduction number
# gen_time = 6.5 #Generation time (in days)
# gt_var = 2.1
# δ = 0.08 #Death rate
# I0 = 10 #initial no. of infections
# ndays = 41*7 #epidemic length
# ρ, ρvar = 0.25, 0.0 #Under reporting, mean/variance
# repd_mean, del_disp = 10.5, 5.0 #Reporting delay, mean/variance

# #Define the parameters of the epidemic
#Disease B (Ebola-like)

R0 = 2.5 #Basic Reproduction number
gen_time = 15 #Generation time (in days)
gt_var = 2.1
δ = 0.08 #Death rate
I0 = 10 #initial no. of infections
ndays = 41*7 #epidemic length
ρ, ρvar = 0.25, 0. #Under reporting, mean/variance
repd_mean, del_disp = 10.5, 5.0 #Reporting delay, mean/variance

rho_beta_a = 8.0
rho_beta_b = (1-ρ)/ρ * rho_beta_a
N = 1e6 #Total population

# Setting-up the control (using non-pharmaceutical interventions)
ctrl_states = ["No restrictions", "Social distancing", "Lockdown"]
R_coeff = [1.0, 0.5, 0.2] #R0_act = R0 * ctrl_states
I_min = 0 #minimum treshold for action

#Sim and control options
# cost_sel = 1 #1: bilinear+oveshoot, 2: flat+quadratic for overshoot
# use_inc = 1 #1: control for incidence, 0: control for infectiousness
# delay_calc_v = 0 #1: as in ref, 0: from incidence
# under_rep_calc = 1 #1: as in ref, 0: separately, from incidence, 2: same, but using a beta distribution for rho
# distr_sel = 1 #0: Deterministic, 1: Poisson, 2: Binomial
# delay_on = 0 #1: sim with time-delay, 0: no-delay (if 0, set delay_calc_v = 0)
# under_rep_on = 0 #0: no under reporting, 1: calculate with under-reporting

# # for ur only
# cost_sel = 1 #1: bilinear+oveshoot, 2: flat+quadratic for overshoot
# use_inc = 1 #1: control for incidence, 0: control for infectiousness
# delay_calc_v = 0 #1: as in ref, 0: from incidence
# under_rep_calc = 3 #1: as in ref, 0: separately, from incidence, 2: same, but using a beta distribution for rho
# distr_sel = 1 #0: Deterministic, 1: Poisson, 2: Binomial
# delay_on = 0 #1: sim with time-delay, 0: no-delay (if 0, set delay_calc_v = 0)
# under_rep_on = 1 #0: no under reporting, 1: calculate with under-reporting

# # for delay only
# cost_sel = 1 #1: bilinear+oveshoot, 2: flat+quadratic for overshoot
# use_inc = 1 #1: control for incidence, 0: control for infectiousness
# delay_calc_v = 0 #1: as in ref, 0: from incidence
# under_rep_calc = 1 #1: as in ref, 0: separately, from incidence, 2: same, but using a beta distribution for rho
# distr_sel = 1 #0: Deterministic, 1: Poisson, 2: Binomial
# delay_on = 1 #1: sim with time-delay, 0: no-delay (if 0, set delay_calc_v = 0)
# under_rep_on = 0 #0: no under reporting, 1: calculate with under-reporting

# # for delay and ur
cost_sel = 1 #1: bilinear+oveshoot, 2: flat+quadratic for overshoot
use_inc = 1 #1: control for incidence, 0: control for infectiousness
delay_calc_v = 0 #1: as in ref, 0: from incidence
under_rep_calc = 3 #1: as in ref, 0: separately, from incidence, 2: same, but using a beta distribution for rho
distr_sel = 1 #0: Deterministic, 1: Poisson, 2: Binomial
delay_on = 1 #1: sim with time-delay, 0: no-delay (if 0, set delay_calc_v = 0)
under_rep_on = 1 #0: no under reporting, 1: calculate with under-reporting

binn = 10 
#est_R = 1 #1: when making predictions use R estimated from data. 0: use model R

#The cost function
cost_of_state = [0.0, 0.01, 0.15] #Cost of interventions/day
Lc_target = 5000 #desired infectiousness
Lc_target_pen = Lc_target*1.5 #extra overshoot penalty
R_target = 1.0
#alpha = 1.3/Lc_target #~proportional gain (regulates error) covid
alpha = 3.25/Lc_target #~proportional gain (regulates error) ebola
beta = 0.0 #~derivative gain (regulates error velocity)
ovp = 5.0 #overshoot penalty
γ = 0.95 #discounting factor

#Simulation parameters
n_ens = 100 #MC assembly size for 4
sim_ens = 100 #assembly size for full simulation

#Frequecy of policy review
rf = 14 #days 7
R_est_wind = 5#rf-2 #window for R estimation
use_S = 0

#Prediction window
pred_days = 14 #14 #21 #12
days = 1:ndays+pred_days

#Distribution of the reporting delay

Ydel = Gamma(del_disp, repd_mean/del_disp)
Ydelpdf = pdf.(Ydel,days)
cdelpdf = sum(Ydelpdf)
nYdel = Ydelpdf/cdelpdf

plt.plot(days,nYdel)
plt.xlim([0,25])
plt.show()

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
under_rep_calc,
under_rep_on,
delay_on,
rho_beta_a,
rho_beta_b)

reward = reward_sel(cost_sel)

#Initialisation
Ivect = zeros(ndays+pred_days, sim_ens)
Svect = zeros(ndays+pred_days, sim_ens) #Susceptibles
Revect = zeros(ndays+pred_days, sim_ens) #Effective Reproduction numer
Rewvect = zeros(ndays+pred_days, sim_ens) #Effective Reproduction numer -- weighted
Lvect = zeros(ndays+pred_days, sim_ens) #Total infectiousness
Lcvect = zeros(ndays+pred_days, sim_ens) #Total infectiousness with perceived infection numbers -- control-basis
cvect = zeros(ndays+pred_days, sim_ens) #Reported cases
Dvect = zeros(ndays+pred_days, sim_ens) #Deceased
Ldvect = zeros(ndays+pred_days, sim_ens)
Rest = zeros(ndays+pred_days, sim_ens) #estimated Re from data 
R0est = zeros(ndays+pred_days, sim_ens) #estimated R0 from data 

#the goverment's policy on handling the epidemic
policy = ones(Int8, ndays+pred_days, sim_ens) #we start without any restrictions

Y = Gamma(gen_time/gt_var, gt_var)

Ypdf = pdf.(Y,days)
cpdf = sum(Ypdf)
nY = Ypdf/cpdf

plt.plot(days,nY)
plt.xlim([0,25])
plt.show()

#Risk of death wrt. time of becoming infectious

Yd = Erlang(7, 3.0)

Ydpdf = pdf.(Yd,days)
cdpdf = sum(Ydpdf)
nYd = Ydpdf/cdpdf

#Initialisation
Ivect[1,:] .= I0
Lvect[1,:] .= I0
Dvect[1,:] .= 0
cvect[1,:] .= ρ*I0
Lcvect[1,:] .= cvect[1]
Svect[1,:] .= N-I0
Revect[1,:] .= R0
Rewvect[1,:] .= R0
Rest[1,:] .= 1.0
R0est[1,:] .= 1.0

#Perform the simulation
Threads.@threads for ii in 1:sim_ens
    #repd_ii = repd[ii]
    Ivect[:,ii], Revect[:,ii], Lvect[:,ii], Lcvect[:,ii], Dvect[:,ii], Svect[:,ii], cvect[:,ii], Rewvect[:,ii], policy[:,ii], Rest[:,ii], R0est[:,ii] = EpiRun_preds_noS(Ivect[:,ii], Revect[:,ii], Lvect[:,ii], Lcvect[:,ii], Ldvect[:,ii], Dvect[:,ii], Svect[:,ii], cvect[:,ii], Rewvect[:,ii], policy[:,ii], Rest[:,ii], R0est[:,ii], nY, nYd, reward, par)
end

## Save results
using JLD

##filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(mm)_delv_$(jj)_under_$(kk).jld"   
#filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_delay_under_toplot.jld"

#@save filename par Ivect Revect Lvect Lcvect Dvect Svect cvect Rewvect policy Rest R0est


## Plot results

low_quant_I = zeros(ndays)
low_quant_L = zeros(ndays)
high_quant_I = zeros(ndays)
high_quant_L = zeros(ndays)
median_quant_I = zeros(ndays)
median_quant_L = zeros(ndays)
mean_I = zeros(ndays)
mean_L = zeros(ndays)

for ii in 1:ndays
    low_quant_I[ii] = quantile(cvect[ii,:], 0.05)
    low_quant_L[ii] = quantile(Lvect[ii,:], 0.05)
    high_quant_I[ii] = quantile(cvect[ii,:], 0.95)
    high_quant_L[ii] = quantile(Lvect[ii,:], 0.95)
    median_quant_I[ii] = quantile(cvect[ii,:], 0.5)
    median_quant_L[ii] = quantile(Lvect[ii,:], 0.5)
    mean_I[ii] = mean(cvect[ii,:])
    mean_L[ii] = mean(Lvect[ii,:])
end

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    L_tmp = Lvect[:,ii]
    p_tmp = policy[:,ii]

    x_n = collect(float(copy(x)))
    x_s = collect(float(copy(x)))
    x_l = collect(float(copy(x)))
    L_n = copy(L_tmp)
    L_s = copy(L_tmp)
    L_l = copy(L_tmp)

    x_n[p_tmp.==2] .= NaN
    x_n[p_tmp.==3] .= NaN
    L_n[p_tmp.==2] .= NaN
    L_n[p_tmp.==3] .= NaN

    x_s[p_tmp.==1] .= NaN
    x_s[p_tmp.==3] .= NaN
    L_s[p_tmp.==1] .= NaN
    L_s[p_tmp.==3] .= NaN

    x_l[p_tmp.==1] .= NaN
    x_l[p_tmp.==2] .= NaN
    L_l[p_tmp.==1] .= NaN
    L_l[p_tmp.==2] .= NaN

    plt.plot(x_n, L_n, color="green", alpha=0.15)
    plt.plot(x_s, L_s, color="purple", alpha=0.15)
    plt.plot(x_l, L_l, color="red", alpha=0.15)
end

#plot the target line
x_targ = [0,ndays]
y_targ = [Lc_target,Lc_target]
y_min = [I_min,I_min]

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.plot(x_targ, y_min, color="blue", linewidth=1.0, linestyle="--")
plt.plot(collect(1:ndays), low_quant_L, color="yellow", linewidth=0.75, linestyle="dotted")
plt.plot(collect(1:ndays), high_quant_L, color="yellow", linewidth=0.75, linestyle="dotted")
plt.plot(collect(1:ndays), median_quant_L, color="yellow", linewidth=0.75, linestyle="dotted")
plt.plot(collect(1:ndays), mean_L, color="cyan", linewidth=0.75, linestyle="dashed")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Total infectiousness")


plt.show()

##show a randomly selected trajectory
ens_sel = rand(1:sim_ens)

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    L_tmp = Lvect[:,ii]
    p_tmp = policy[:,ii]

    x_n = collect(float(copy(x)))
    x_s = collect(float(copy(x)))
    x_l = collect(float(copy(x)))
    L_n = copy(L_tmp)
    L_s = copy(L_tmp)
    L_l = copy(L_tmp)

    x_n[p_tmp.==2] .= NaN
    x_n[p_tmp.==3] .= NaN
    L_n[p_tmp.==2] .= NaN
    L_n[p_tmp.==3] .= NaN

    x_s[p_tmp.==1] .= NaN
    x_s[p_tmp.==3] .= NaN
    L_s[p_tmp.==1] .= NaN
    L_s[p_tmp.==3] .= NaN

    x_l[p_tmp.==1] .= NaN
    x_l[p_tmp.==2] .= NaN
    L_l[p_tmp.==1] .= NaN
    L_l[p_tmp.==2] .= NaN

    plt.plot(x_n, L_n, color="lightgreen", alpha=0.005)
    plt.plot(x_s, L_s, color="magenta", alpha=0.003)
    plt.plot(x_l, L_l, color="red", alpha=0.005)
end

x = 1:ndays+pred_days
L_tmp = Lvect[:,ens_sel]
p_tmp = policy[:,ens_sel]

x_n = collect(float(copy(x)))
x_s = collect(float(copy(x)))
x_l = collect(float(copy(x)))
L_n = copy(L_tmp)
L_s = copy(L_tmp)
L_l = copy(L_tmp)

x_n[p_tmp.==2] .= NaN
x_n[p_tmp.==3] .= NaN
L_n[p_tmp.==2] .= NaN
L_n[p_tmp.==3] .= NaN

x_s[p_tmp.==1] .= NaN
x_s[p_tmp.==3] .= NaN
L_s[p_tmp.==1] .= NaN
L_s[p_tmp.==3] .= NaN

x_l[p_tmp.==1] .= NaN
x_l[p_tmp.==2] .= NaN
L_l[p_tmp.==1] .= NaN
L_l[p_tmp.==2] .= NaN

plt.plot(x_n, L_n, color="green", linewidth=3, alpha=1)
plt.plot(x_s, L_s, color="purple", linewidth=3, alpha=1)
plt.plot(x_l, L_l, color="red", linewidth=3, alpha=1)

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.plot(x_targ, y_min, color="blue", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Total infectiousness")

plt.show()
#plt.savefig("lambda_idR.png", dpi=300)

## Plot results

low_quant_I = zeros(ndays)
low_quant_L = zeros(ndays)
high_quant_I = zeros(ndays)
high_quant_L = zeros(ndays)
median_quant_I = zeros(ndays)
median_quant_L = zeros(ndays)
mean_I = zeros(ndays)
mean_L = zeros(ndays)

for ii in 1:ndays
    low_quant_I[ii] = quantile(cvect[ii,:], 0.05)
    low_quant_L[ii] = quantile(Lcvect[ii,:], 0.05)
    high_quant_I[ii] = quantile(cvect[ii,:], 0.95)
    high_quant_L[ii] = quantile(Lcvect[ii,:], 0.95)
    median_quant_I[ii] = quantile(cvect[ii,:], 0.5)
    median_quant_L[ii] = quantile(Lcvect[ii,:], 0.5)
    mean_I[ii] = mean(cvect[ii,:])
    mean_L[ii] = mean(Lcvect[ii,:])
end

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    L_tmp = Lcvect[:,ii]
    p_tmp = policy[:,ii]

    x_n = collect(float(copy(x)))
    x_s = collect(float(copy(x)))
    x_l = collect(float(copy(x)))
    L_n = copy(L_tmp)
    L_s = copy(L_tmp)
    L_l = copy(L_tmp)

    x_n[p_tmp.==2] .= NaN
    x_n[p_tmp.==3] .= NaN
    L_n[p_tmp.==2] .= NaN
    L_n[p_tmp.==3] .= NaN

    x_s[p_tmp.==1] .= NaN
    x_s[p_tmp.==3] .= NaN
    L_s[p_tmp.==1] .= NaN
    L_s[p_tmp.==3] .= NaN

    x_l[p_tmp.==1] .= NaN
    x_l[p_tmp.==2] .= NaN
    L_l[p_tmp.==1] .= NaN
    L_l[p_tmp.==2] .= NaN

    plt.plot(x_n, L_n, color="green", alpha=0.15)
    plt.plot(x_s, L_s, color="purple", alpha=0.15)
    plt.plot(x_l, L_l, color="red", alpha=0.15)
end

#plot the target line
x_targ = [0,ndays]
y_targ = [Lc_target,Lc_target]
y_min = [I_min,I_min]

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.plot(x_targ, y_min, color="blue", linewidth=1.0, linestyle="--")
plt.plot(collect(1:ndays), low_quant_L, color="yellow", linewidth=0.75, linestyle="dotted")
plt.plot(collect(1:ndays), high_quant_L, color="yellow", linewidth=0.75, linestyle="dotted")
plt.plot(collect(1:ndays), median_quant_L, color="yellow", linewidth=0.75, linestyle="dotted")
plt.plot(collect(1:ndays), mean_L, color="cyan", linewidth=0.75, linestyle="dashed")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Total infectiousness (from rep. cases)")


plt.show()

##show a randomly selected trajectory
ens_sel = rand(1:sim_ens)

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    L_tmp = Lcvect[:,ii]
    p_tmp = policy[:,ii]

    x_n = collect(float(copy(x)))
    x_s = collect(float(copy(x)))
    x_l = collect(float(copy(x)))
    L_n = copy(L_tmp)
    L_s = copy(L_tmp)
    L_l = copy(L_tmp)

    x_n[p_tmp.==2] .= NaN
    x_n[p_tmp.==3] .= NaN
    L_n[p_tmp.==2] .= NaN
    L_n[p_tmp.==3] .= NaN

    x_s[p_tmp.==1] .= NaN
    x_s[p_tmp.==3] .= NaN
    L_s[p_tmp.==1] .= NaN
    L_s[p_tmp.==3] .= NaN

    x_l[p_tmp.==1] .= NaN
    x_l[p_tmp.==2] .= NaN
    L_l[p_tmp.==1] .= NaN
    L_l[p_tmp.==2] .= NaN

    plt.plot(x_n, L_n, color="lightgreen", alpha=0.005)
    plt.plot(x_s, L_s, color="magenta", alpha=0.003)
    plt.plot(x_l, L_l, color="red", alpha=0.005)
end

x = 1:ndays+pred_days
L_tmp = Lcvect[:,ens_sel]
p_tmp = policy[:,ens_sel]

x_n = collect(float(copy(x)))
x_s = collect(float(copy(x)))
x_l = collect(float(copy(x)))
L_n = copy(L_tmp)
L_s = copy(L_tmp)
L_l = copy(L_tmp)

x_n[p_tmp.==2] .= NaN
x_n[p_tmp.==3] .= NaN
L_n[p_tmp.==2] .= NaN
L_n[p_tmp.==3] .= NaN

x_s[p_tmp.==1] .= NaN
x_s[p_tmp.==3] .= NaN
L_s[p_tmp.==1] .= NaN
L_s[p_tmp.==3] .= NaN

x_l[p_tmp.==1] .= NaN
x_l[p_tmp.==2] .= NaN
L_l[p_tmp.==1] .= NaN
L_l[p_tmp.==2] .= NaN

plt.plot(x_n, L_n, color="green", linewidth=3, alpha=1)
plt.plot(x_s, L_s, color="purple", linewidth=3, alpha=1)
plt.plot(x_l, L_l, color="red", linewidth=3, alpha=1)

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.plot(x_targ, y_min, color="blue", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Total infectiousness (from rep. cases)")

plt.show()
#plt.savefig("lambda_idR.png", dpi=300)

##show incidence

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    I_tmp = cvect[:,ii]

    p_tmp = policy[:,ii]

    x_n = collect(float(copy(x)))
    x_s = collect(float(copy(x)))
    x_l = collect(float(copy(x)))
    I_n = copy(I_tmp)
    I_s = copy(I_tmp)
    I_l = copy(I_tmp)

    x_n[p_tmp.==2] .= NaN
    x_n[p_tmp.==3] .= NaN
    I_n[p_tmp.==2] .= NaN
    I_n[p_tmp.==3] .= NaN

    x_s[p_tmp.==1] .= NaN
    x_s[p_tmp.==3] .= NaN
    I_s[p_tmp.==1] .= NaN
    I_s[p_tmp.==3] .= NaN

    x_l[p_tmp.==1] .= NaN
    x_l[p_tmp.==2] .= NaN
    I_l[p_tmp.==1] .= NaN
    I_l[p_tmp.==2] .= NaN

    plt.plot(x_n, I_n, color="green", alpha=0.15)
    plt.plot(x_s, I_s, color="purple", alpha=0.15)
    plt.plot(x_l, I_l, color="red", alpha=0.15)
end

#plot the target line
x_targ = [0,ndays]
y_targ = [Lc_target,Lc_target]
y_min = [I_min,I_min]

min_I_vect = zeros(sim_ens)
max_I_vect = zeros(sim_ens)
bound_starts = zeros(sim_ens)

# for ii in 1:sim_ens

#     R_cross_one = diff(sign.(Rewvect[:,ii].-1.0)).<0
#     indxs = 1:length(R_cross_one)

#     indx_ss = indxs[R_cross_one][1]
#     bound_starts[ii] = indx_ss

#     reduced_I = Ivect[indx_ss:ndays, ii]

#     I_cross_targ = diff(sign.(reduced_I.-Lc_target)).<0
#     indxs_I = 1:length(I_cross_targ)

#     indx_ss_I = indxs_I[I_cross_targ][1]
#     bound_starts[ii] = indx_ss + indx_ss_I + 1
    
#     max_I_vect[ii] = maximum(Ivect[(indx_ss + indx_ss_I):ndays, ii])
#     min_I_vect[ii] = minimum(Ivect[(indx_ss + indx_ss_I):ndays, ii])

# end

bound_start = minimum(bound_starts)
env_size = max_I_vect - min_I_vect

max100 = maximum(max_I_vect)
max95 = quantile(max_I_vect, 0.95)
min5 = quantile(min_I_vect, 0.05)
min0 = minimum(min_I_vect)

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
#plt.plot(x_targ, y_min, color="blue", linewidth=1.0, linestyle="--")
#plt.plot(collect(1:ndays), low_quant_I, color="yellow", linewidth=0.75, linestyle="dotted")
#plt.plot(collect(1:ndays), high_quant_I, color="yellow", linewidth=0.75, linestyle="dotted")
#plt.plot(collect(1:ndays), median_quant_I, color="yellow", linewidth=0.75, linestyle="dotted")
#plt.plot(collect(1:ndays), mean_I, color="cyan", linewidth=0.75, linestyle="dashed")

#plt.plot([bound_start,ndays], [max100,max100], color="red", linewidth=2.0, linestyle="--")
#plt.plot([bound_start,ndays], [min0,min0], color="red", linewidth=2.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("New infections")

#cd("figs4/ebola_real14_bin")
#plt.savefig("incidence_multi.svg")
plt.show()

## highlight one trajectory

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    I_tmp = cvect[:,ii]
    p_tmp = policy[:,ii]

    # x_n = collect(float(copy(x)))
    # x_s = collect(float(copy(x)))
    # x_l = collect(float(copy(x)))
    # I_n = copy(I_tmp)
    # I_s = copy(I_tmp)
    # I_l = copy(I_tmp)

    # x_n[p_tmp.==2] .= NaN
    # x_n[p_tmp.==3] .= NaN
    # I_n[p_tmp.==2] .= NaN
    # I_n[p_tmp.==3] .= NaN

    # x_s[p_tmp.==1] .= NaN
    # x_s[p_tmp.==3] .= NaN
    # I_s[p_tmp.==1] .= NaN
    # I_s[p_tmp.==3] .= NaN

    # x_l[p_tmp.==1] .= NaN
    # x_l[p_tmp.==2] .= NaN
    # I_l[p_tmp.==1] .= NaN
    # I_l[p_tmp.==2] .= NaN

    # plt.plot(x_n, I_n, color="lightgreen", alpha=0.05/1.5)
    # plt.plot(x_s, I_s, color="magenta", alpha=0.03/1.5)
    plt.plot(x, I_tmp, color="black", alpha=0.05/1.5)

end

x = 1:ndays+pred_days
I_tmp = cvect[:,ens_sel]
I2_tmp = Ivect[:,ens_sel]
p_tmp = policy[:,ens_sel]

x_n = collect(float(copy(x)))
x_s = collect(float(copy(x)))
x_l = collect(float(copy(x)))
I_n = copy(I_tmp)
I_s = copy(I_tmp)
I_l = copy(I_tmp)

I2_n = copy(I2_tmp)
I2_s = copy(I2_tmp)
I2_l = copy(I2_tmp)

x_n[p_tmp.==2] .= NaN
x_n[p_tmp.==3] .= NaN
I_n[p_tmp.==2] .= NaN
I_n[p_tmp.==3] .= NaN
I2_n[p_tmp.==2] .= NaN
I2_n[p_tmp.==3] .= NaN

x_s[p_tmp.==1] .= NaN
x_s[p_tmp.==3] .= NaN
I_s[p_tmp.==1] .= NaN
I_s[p_tmp.==3] .= NaN
I2_s[p_tmp.==1] .= NaN
I2_s[p_tmp.==3] .= NaN

x_l[p_tmp.==1] .= NaN
x_l[p_tmp.==2] .= NaN
I_l[p_tmp.==1] .= NaN
I_l[p_tmp.==2] .= NaN
I2_l[p_tmp.==1] .= NaN
I2_l[p_tmp.==2] .= NaN

plt.plot(collect(1:ndays), low_quant_I, color="black", linewidth=1.0)
plt.plot(collect(1:ndays), high_quant_I, color="black", linewidth=1.0)
#plt.plot(collect(1:ndays), median_quant_I, color="yellow", linewidth=0.75, linestyle="dotted")
plt.plot(collect(1:ndays), mean_I, color="black", linewidth=1.0)

plt.plot(x_n, I_n, color="green", linewidth=3, alpha=1)
plt.plot(x_s, I_s, color="purple", linewidth=3, alpha=1)
plt.plot(x_l, I_l, color="red", linewidth=3, alpha=1)

plt.plot(x_n, I2_n, color="green", linewidth=1, alpha=1, linestyle="-")
plt.plot(x_s, I2_s, color="purple", linewidth=1, alpha=1, linestyle="-")
plt.plot(x_l, I2_l, color="red", linewidth=1, alpha=1, linestyle="-")

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
#plt.plot(x_targ, y_min, color="blue", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("New cases/infections")

plt.savefig("incidence_highlight.svg")
plt.show()
#plt.savefig("I_idR.png", dpi=300)

##show reported cases

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    c_tmp = cvect[:,ii]

    p_tmp = policy[:,ii]

    x_n = collect(float(copy(x)))
    x_s = collect(float(copy(x)))
    x_l = collect(float(copy(x)))
    c_n = copy(c_tmp)
    c_s = copy(c_tmp)
    c_l = copy(c_tmp)

    x_n[p_tmp.==2] .= NaN
    x_n[p_tmp.==3] .= NaN
    c_n[p_tmp.==2] .= NaN
    c_n[p_tmp.==3] .= NaN

    x_s[p_tmp.==1] .= NaN
    x_s[p_tmp.==3] .= NaN
    c_s[p_tmp.==1] .= NaN
    c_s[p_tmp.==3] .= NaN

    x_l[p_tmp.==1] .= NaN
    x_l[p_tmp.==2] .= NaN
    c_l[p_tmp.==1] .= NaN
    c_l[p_tmp.==2] .= NaN

    plt.plot(x_n, c_n, color="green", alpha=0.15)
    plt.plot(x_s, c_s, color="purple", alpha=0.15)
    plt.plot(x_l, c_l, color="red", alpha=0.15)
end

#plot the target line
x_targ = [0,ndays]
y_targ = [Lc_target,Lc_target]
y_min = [I_min,I_min]

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.plot(x_targ, y_min, color="blue", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Reported cases")


plt.show()

## highlight one trajectory

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    c_tmp = cvect[:,ii]
    p_tmp = policy[:,ii]

    x_n = collect(float(copy(x)))
    x_s = collect(float(copy(x)))
    x_l = collect(float(copy(x)))
    c_n = copy(c_tmp)
    c_s = copy(c_tmp)
    c_l = copy(c_tmp)

    x_n[p_tmp.==2] .= NaN
    x_n[p_tmp.==3] .= NaN
    c_n[p_tmp.==2] .= NaN
    c_n[p_tmp.==3] .= NaN

    x_s[p_tmp.==1] .= NaN
    x_s[p_tmp.==3] .= NaN
    c_s[p_tmp.==1] .= NaN
    c_s[p_tmp.==3] .= NaN

    x_l[p_tmp.==1] .= NaN
    x_l[p_tmp.==2] .= NaN
    c_l[p_tmp.==1] .= NaN
    c_l[p_tmp.==2] .= NaN

    plt.plot(x_n, c_n, color="lightgreen", alpha=0.005)
    plt.plot(x_s, c_s, color="magenta", alpha=0.003)
    plt.plot(x_l, c_l, color="red", alpha=0.005)
end

x = 1:ndays+pred_days
c_tmp = cvect[:,ens_sel]
p_tmp = policy[:,ens_sel]

x_n = collect(float(copy(x)))
x_s = collect(float(copy(x)))
x_l = collect(float(copy(x)))
c_n = copy(c_tmp)
c_s = copy(c_tmp)
c_l = copy(c_tmp)

x_n[p_tmp.==2] .= NaN
x_n[p_tmp.==3] .= NaN
c_n[p_tmp.==2] .= NaN
c_n[p_tmp.==3] .= NaN

x_s[p_tmp.==1] .= NaN
x_s[p_tmp.==3] .= NaN
c_s[p_tmp.==1] .= NaN
c_s[p_tmp.==3] .= NaN

x_l[p_tmp.==1] .= NaN
x_l[p_tmp.==2] .= NaN
c_l[p_tmp.==1] .= NaN
c_l[p_tmp.==2] .= NaN

plt.plot(x_n, c_n, color="green", linewidth=3, alpha=1)
plt.plot(x_s, c_s, color="purple", linewidth=3, alpha=1)
plt.plot(x_l, c_l, color="red", linewidth=3, alpha=1)

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.plot(x_targ, y_min, color="blue", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Reported cases")


plt.show()

##show reproduction number 

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    R_tmp = Rewvect[:,ii]
    p_tmp = policy[:,ii]

    x_n = collect(float(copy(x)))
    x_s = collect(float(copy(x)))
    x_l = collect(float(copy(x)))
    R_n = copy(R_tmp)
    R_s = copy(R_tmp)
    R_l = copy(R_tmp)

    x_n[p_tmp.==2] .= NaN
    x_n[p_tmp.==3] .= NaN
    R_n[p_tmp.==2] .= NaN
    R_n[p_tmp.==3] .= NaN

    x_s[p_tmp.==1] .= NaN
    x_s[p_tmp.==3] .= NaN
    R_s[p_tmp.==1] .= NaN
    R_s[p_tmp.==3] .= NaN

    x_l[p_tmp.==1] .= NaN
    x_l[p_tmp.==2] .= NaN
    R_l[p_tmp.==1] .= NaN
    R_l[p_tmp.==2] .= NaN

    plt.plot(x_n, R_n, color="green", alpha=0.15)
    plt.plot(x_s, R_s, color="purple", alpha=0.15)
    plt.plot(x_l, R_l, color="red", alpha=0.15)
end

#plot the target line
x_targ = [0,ndays]
y_targ = [1.0,1.0]
#y_min = [I_min,I_min]

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Reproduction number")


plt.show()

## Highlight one trajectory

for ii in 1:sim_ens
    x = 1:ndays+pred_days
    R_tmp = Rewvect[:,ii]
    p_tmp = policy[:,ii]

    # x_n = collect(float(copy(x)))
    # x_s = collect(float(copy(x)))
    # x_l = collect(float(copy(x)))
    # R_n = copy(R_tmp)
    # R_s = copy(R_tmp)
    # R_l = copy(R_tmp)

    # x_n[p_tmp.==2] .= NaN
    # x_n[p_tmp.==3] .= NaN
    # R_n[p_tmp.==2] .= NaN
    # R_n[p_tmp.==3] .= NaN

    # x_s[p_tmp.==1] .= NaN
    # x_s[p_tmp.==3] .= NaN
    # R_s[p_tmp.==1] .= NaN
    # R_s[p_tmp.==3] .= NaN

    # x_l[p_tmp.==1] .= NaN
    # x_l[p_tmp.==2] .= NaN
    # R_l[p_tmp.==1] .= NaN
    # R_l[p_tmp.==2] .= NaN

    # plt.plot(x_n, R_n, color="lightgreen", alpha=0.05/1.5)
    # plt.plot(x_s, R_s, color="magenta", alpha=0.03/1.5)
    plt.plot(x, R_tmp, color="black", alpha=0.05/1.5)
end

x = 1:ndays+pred_days
R_tmp = Rewvect[:,ens_sel]
p_tmp = policy[:,ens_sel]
R_est_tmp = Rest[:,ens_sel]

x_n = collect(float(copy(x)))
x_s = collect(float(copy(x)))
x_l = collect(float(copy(x)))
R_n = copy(R_tmp)
R_s = copy(R_tmp)
R_l = copy(R_tmp)

R_est_n = copy(R_est_tmp)
R_est_s = copy(R_est_tmp)
R_est_l = copy(R_est_tmp)

x_n[p_tmp.==2] .= NaN
x_n[p_tmp.==3] .= NaN
R_n[p_tmp.==2] .= NaN
R_n[p_tmp.==3] .= NaN
R_est_n[p_tmp.==2] .= NaN
R_est_n[p_tmp.==3] .= NaN

x_s[p_tmp.==1] .= NaN
x_s[p_tmp.==3] .= NaN
R_s[p_tmp.==1] .= NaN
R_s[p_tmp.==3] .= NaN
R_est_s[p_tmp.==1] .= NaN
R_est_s[p_tmp.==3] .= NaN

x_l[p_tmp.==1] .= NaN
x_l[p_tmp.==2] .= NaN
R_l[p_tmp.==1] .= NaN
R_l[p_tmp.==2] .= NaN
R_est_l[p_tmp.==1] .= NaN
R_est_l[p_tmp.==2] .= NaN

plt.plot(x_n, R_n, color="green", linewidth=3, alpha=1)
plt.plot(x_s, R_s, color="purple", linewidth=3, alpha=1)
plt.plot(x_l, R_l, color="red", linewidth=3, alpha=1)

plt.plot(x_n, R_est_n, color="green", linewidth=1, alpha=1, linestyle="-")
plt.plot(x_s, R_est_s, color="purple", linewidth=1, alpha=1, linestyle="-")
plt.plot(x_l, R_est_l, color="red", linewidth=1, alpha=1, linestyle="-")

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Reproduction number")
plt.ylim([0,4])

plt.savefig("R_highlight.svg")
plt.show()


## Plot the actual reproduction number against the estimated one

x = 1:ndays+pred_days
R_tmp = Rewvect[:,ens_sel]
R_est_tmp = Rest[:,ens_sel]
p_tmp = policy[:,ens_sel]

x_n = collect(float(copy(x)))
x_s = collect(float(copy(x)))
x_l = collect(float(copy(x)))
R_n = copy(R_tmp)
R_s = copy(R_tmp)
R_l = copy(R_tmp)
R_est_n = copy(R_est_tmp)
R_est_s = copy(R_est_tmp)
R_est_l = copy(R_est_tmp)

x_n[p_tmp.==2] .= NaN
x_n[p_tmp.==3] .= NaN
R_n[p_tmp.==2] .= NaN
R_n[p_tmp.==3] .= NaN
R_est_n[p_tmp.==2] .= NaN
R_est_n[p_tmp.==3] .= NaN

x_s[p_tmp.==1] .= NaN
x_s[p_tmp.==3] .= NaN
R_s[p_tmp.==1] .= NaN
R_s[p_tmp.==3] .= NaN
R_est_s[p_tmp.==1] .= NaN
R_est_s[p_tmp.==3] .= NaN

x_l[p_tmp.==1] .= NaN
x_l[p_tmp.==2] .= NaN
R_l[p_tmp.==1] .= NaN
R_l[p_tmp.==2] .= NaN
R_est_l[p_tmp.==1] .= NaN
R_est_l[p_tmp.==2] .= NaN

plt.plot(x_n, R_n, color="green", linewidth=2, alpha=1)
plt.plot(x_s, R_s, color="purple", linewidth=2, alpha=1)
plt.plot(x_l, R_l, color="red", linewidth=2, alpha=1)
plt.plot(x_n, R_est_n, color="green", linewidth=2, alpha=1, linestyle="--")
plt.plot(x_s, R_est_s, color="purple", linewidth=2, alpha=1, linestyle="--")
plt.plot(x_l, R_est_l, color="red", linewidth=2, alpha=1, linestyle="--")

review_days = collect(rf:rf:ndays)
R_est_at_rew = R_est_tmp[review_days]
R_at_rew = R_tmp[review_days]
plt.scatter(review_days, R_est_at_rew, color="blue")
plt.scatter(review_days, R_at_rew, color="black")

plt.plot(x_targ, y_targ, color="black", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Reproduction number")


plt.show()
#plt.savefig("R_act_vs_est.png", dpi=300)

## and now the estimated R0

x = 1:ndays+pred_days

R0_tmp = R0est[:,ens_sel]
p_tmp = policy[:,ens_sel]

x_n = collect(float(copy(x)))
x_s = collect(float(copy(x)))
x_l = collect(float(copy(x)))
R0_est_n = copy(R0_tmp)
R0_est_s = copy(R0_tmp)
R0_est_l = copy(R0_tmp)

x_n[p_tmp.==2] .= NaN
x_n[p_tmp.==3] .= NaN
R0_est_n[p_tmp.==2] .= NaN
R0_est_n[p_tmp.==3] .= NaN

x_s[p_tmp.==1] .= NaN
x_s[p_tmp.==3] .= NaN
R0_est_s[p_tmp.==1] .= NaN
R0_est_s[p_tmp.==3] .= NaN

x_l[p_tmp.==1] .= NaN
x_l[p_tmp.==2] .= NaN
R0_est_l[p_tmp.==1] .= NaN
R0_est_l[p_tmp.==2] .= NaN

plt.plot(x_n, R0_est_n, color="green", linewidth=2, alpha=1)
plt.plot(x_s, R0_est_s, color="purple", linewidth=2, alpha=1)
plt.plot(x_l, R0_est_l, color="red", linewidth=2, alpha=1)

review_days = collect(rf:rf:ndays)
R0_est_at_rew = R0_tmp[review_days]
plt.scatter(review_days, R0_est_at_rew, color="blue")

plt.plot(x_targ, [R0, R0], color="black", linewidth=1.0, linestyle="--")
plt.xlim([0,ndays])
plt.xlabel("time [days]")
plt.ylabel("Reproduction number")


plt.show()

## Policy stats
policy

n_vect = zeros(Int, sim_ens)
s_vect = zeros(Int, sim_ens)
l_vect = zeros(Int, sim_ens)

for ii in 1:sim_ens

    n_vect[ii] = count(x->(x.==1), policy[1:ndays,ii])
    s_vect[ii] = count(x->(x.==2), policy[1:ndays,ii])
    l_vect[ii] = count(x->(x.==3), policy[1:ndays,ii])

end

# Pie chart on the whole dataset
n_mean = mean(n_vect)
s_mean = mean(s_vect)
l_mean = mean(l_vect)

means = [n_mean, s_mean, l_mean]
plt.pie(means, colors=["green","purple","red"],autopct="%1.1f%%",labels=ctrl_states)
my_circle=plt.Circle( (0,0), 0.7, color="white")
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.savefig("policy_pie.svg")
plt.show()

##