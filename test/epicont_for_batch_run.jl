using EpiCont

using Random, Distributions
using Base.Threads
using PyCall, Conda

using JLD

#Conda.add("seaborn")
plt = pyimport("matplotlib.pyplot")
np = pyimport("numpy")
sns = pyimport("seaborn")

# mean_delays_to_test
delays = 3.5:3.5:21
delay_dispersion = ([0.1, 1.0, 5.0, 25.0, 200.0])
under_reporting = [1.0]


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
# ρ, ρvar = 1.0, 0.0 #Under reporting, mean/variance
# repd_mean, del_disp = 6, 2 #Reporting delay, mean/variance

# #Define the parameters of the epidemic
#Disease B (Ebola-like)

R0 = 2.5 #Basic Reproduction number
gen_time = 15 #Generation time (in days)
gt_var = 2.1
δ = 0.08 #Death rate
I0 = 10 #initial no. of infections
ndays = 41*7 #epidemic length
ρ, ρvar = 1.0, 0. #Under reporting, mean/variance
repd_mean, del_disp = 12, 2 #Reporting delay, mean/variance

rho_beta_a = 150.0
rho_beta_b = (1-ρ)/ρ * rho_beta_a

N = 1e7 #Total population

# Setting-up the control (using non-pharmaceutical interventions)
ctrl_states = ["No restrictions", "Social distancing", "Lockdown"]
R_coeff = [1.0, 0.5, 0.2] #R0_act = R0 * ctrl_states
I_min = 100 #minimum treshold for action

#Sim and control options
cost_sel = 1 #1: bilinear+oveshoot, 2: flat+quadratic for overshoot
use_inc = 1 #1: control for incidence, 0: control for infectiousness
delay_calc_v = 0 #1: as in ref, 0: from incidence
under_rep_calc = 1 #1: as in ref, 0: separately, from incidence, 2: same, but using a beta distribution for rho
distr_sel = 1 #0: Deterministic, 1: Poisson, 2: Binomial
delay_on = 1 #1: sim with time-delay, 0: no-delay (if 0, set delay_calc_v = 0)
under_rep_on = 0 #0: no under reporting, 1: calculate with under-reporting

binn = 10 
#est_R = 1 #1: when making predictions use R estimated from data. 0: use model R

#The cost function
cost_of_state = [0.0, 0.01, 0.15] #Cost of interventions/day
Lc_target = 5000 #desired infectiousness
Lc_target_pen = Lc_target*1.5 #extra overshoot penalty
R_target = 1.0
# alpha = 1.3/Lc_target #~proportional gain (regulates error)
alpha = 3.25/Lc_target #~proportional gain (regulates error)
beta = 0.0 #~derivative gain (regulates error velocity)
cost_of_state = [0.0, 0.01, 0.15]
ovp = 5.0 #overshoot penalty
γ = 0.95 #discounting factor

#Simulation parameters
n_ens = 100 #MC assembly size for prediction
sim_ens = 1000 #assembly size for full simulation

#Frequecy of policy review
rf = 7 #days
R_est_wind = rf-2 #window for R estimation
use_S = 0

#Prediction window
pred_days = 12
days = 1:ndays+pred_days

#Distribution of the reporting delay

Ydel = Gamma(del_disp, repd_mean/del_disp)
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
under_rep_calc,
under_rep_on,
delay_on,
rho_beta_a,
rho_beta_b)

reward = reward_sel(1)

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

for mm in delays
    for jj in delay_dispersion
        for kk in under_reporting

            println("delay: $(mm), delay_disp: $(jj), under_rep: $(kk)")

            ρ = kk

            Ydel = Gamma(jj, mm/jj)
            Ydelpdf = pdf.(Ydel,days)
            cdelpdf = sum(Ydelpdf)
            nYdel = Ydelpdf/cdelpdf
            
            if delay_on == 0
                nYdel = zeros(length(days))
                nYdel[1] = 1.0
            end

            par.nYdel = nYdel
            par.ρ = ρ

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

            filename = "rerun3_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(mm)_delv_$(jj)_under_$(kk)_noS_ebola.jld"

            @save filename par Ivect Revect Lvect Lcvect Dvect Svect cvect Rewvect policy Rest R0est mm jj kk

        end
    end
end

