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
delay_dispersion = [0.1, 1.0, 5.0, 25.0, 200.0]
under_reporting = [1.0]

LDcost = 0.15
SDcost = 0.01

##Plot peak incidence

del_disp = delay_dispersion[2]
ur = under_reporting[1]
del = delays[1]
delv = [del]
del_dispv = [del_disp]

rf = 7
γ = 0.95
ndays = 41*7 #epidemic length
Lc_target = 5000 #desired infectiousness

cost_error_coeff = 0.0

#filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur).jld"
#filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS.jld"
#filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS.jld"
#filename = "rerun_workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_cyclic_3_5_ebola.jld"
#filename = "rerun_workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_new_ebola.jld"
#filename = "rerun_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_ebola.jld"

#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_then_cyclic_9_75.jld"
#filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS.jld"
#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_using_lambda.jld"
#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_ebola.jld"
#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_cyclic_3_1_ebola.jld"
#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_then_cyclic_4_1.jld"
#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_new_ebola.jld"
#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_cyclic_4_1.jld"
#filename = "rerun3_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_covid.jld"
#filename = "rerun3_workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_covid.jld"

filename = "rerun3_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_ebola.jld"
#filename = "rerun3_workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_ebola.jld"
#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_then_cyclic_ebola_21_36.jld"

data = load(filename)

mean_of_peaks = zeros(length(delays))

Ivect = data["Ivect"]
Rewvect = data["Rewvect"]
max_infs = maximum(Ivect[1:ndays,:], dims=1)'
xs = del*ones(length(max_infs))
disp = del_disp*ones(length(max_infs))
mean_of_peaks[1] = mean(max_infs)

policy = data["policy"]
cost_daily = zeros(size(policy))
cost_daily[policy.==3] .= LDcost
cost_daily[policy.==2] .= SDcost

cost_daily[Ivect.>1.5*Lc_target] = cost_daily[Ivect.>1.5*Lc_target] .+ 5.0
sum_daily_costs = 1/ndays*sum(cost_daily, dims=1)'
#sum_daily_costs = sum(cost_daily, dims=1)'

Ns = [sum(policy[1:ndays,:].==1)]
SDs = [sum(policy[1:ndays,:].==2)]
LDs = [sum(policy[1:ndays,:].==3)]

sim_ens=size(Ivect)[2]

xdata = []#string.(xs)
dispdata = []#string.(disp)
peakdata = []#vec(max_infs')
costdata = []#vec(sum_daily_costs')

#plt.scatter(xs, max_infs, color="blue", alpha=0.01)

# add enevelope size
min_I_vect = zeros(sim_ens)
max_I_vect = zeros(sim_ens)
bound_starts = zeros(sim_ens)
ncs = Bool.(ones(length(max_infs)))

for kk in 1:sim_ens

    R_cross_one = diff(sign.(Rewvect[:,kk].-1.0)).<0
    indxs = 1:length(R_cross_one)

    indx_ss = indxs[R_cross_one][1]
    bound_starts[kk] = indx_ss

    reduced_I = Ivect[indx_ss:ndays, kk]

    I_cross_targ = diff(sign.(reduced_I.-Lc_target)).<0
    indxs_I = 1:length(I_cross_targ)

    println(length(indxs_I[I_cross_targ]) < 1)

    if length(indxs_I[I_cross_targ]) < 1
        max_I_vect[kk] = maximum(Ivect[:, kk])#NaN
        min_I_vect[kk] = minimum(Ivect[:, kk])#NaN
        ncs[kk] = 0
    else
        indx_ss_I = indxs_I[I_cross_targ][1]
        bound_starts[kk] = indx_ss + indx_ss_I + 1
        
        max_I_vect[kk] = maximum(Ivect[(indx_ss + indx_ss_I):ndays, kk])
        min_I_vect[kk] = minimum(Ivect[(indx_ss + indx_ss_I):ndays, kk])
    end

end

bound_start = minimum(bound_starts)
env_size = max_I_vect - min_I_vect
envdata = []#vec(env_size')
ncdata = []#vec(ncs')
costs2 = []
overshoot_days = []

for ii in 1:length(delays)
    for jj in 2:length(delay_dispersion)

        del = delays[ii]
        del_disp = delay_dispersion[jj]

        append!(delv,[del])
        append!(del_dispv,[del_disp])

        #filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur).jld"
        #filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS.jld"
        #filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS.jld"
        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_using_lambda.jld"
        #filename = "rerun_workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_cyclic_3_5_ebola.jld"
        #filename = "rerun_workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_new_ebola.jld"
        #filename = "rerun_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_ebola.jld"

        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_then_cyclic_9_75.jld"
        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_ebola.jld"
        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_cyclic_3_1_ebola.jld"
        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_then_cyclic_4_1.jld"
        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_new_ebola.jld"
        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_cyclic_4_1.jld"

        #filename = "rerun3_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_covid.jld"
        #filename = "rerun3_workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_covid.jld"

        filename = "rerun3_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_ebola.jld"
        #filename = "rerun3_workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_ebola.jld"
        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold_then_cyclic_ebola_21_36.jld"
        data = load(filename)

        #Plot peak incidence

        Ivect = data["Ivect"]
        Rewvect = data["Rewvect"]

        policy = data["policy"]

        cost_daily = zeros(size(policy))
        overshoots = zeros(size(policy))

        cost_daily[policy.==3] .= LDcost
        cost_daily[policy.==2] .= SDcost

        cost_daily[Ivect.>1.5*Lc_target] = cost_daily[Ivect.>1.5*Lc_target] .+ 5.0

        overshoots[Ivect.>1.5*Lc_target] .= 1.0

        sum_daily_costs = 1/ndays*sum(cost_daily, dims=1)'

        Ns = append!(Ns,[sum(policy[1:ndays,:].==1)])
        SDs = append!(SDs,[sum(policy[1:ndays,:].==2)])
        LDs = append!(LDs,[sum(policy[1:ndays,:].==3)])

        max_infs = maximum(Ivect[1:ndays,:], dims=1)'
        xs = del*ones(length(max_infs))
        disp = del_disp*ones(length(max_infs))

        mean_of_peaks[ii] = mean(max_infs)

        xdata = append!(xdata,string.(xs))
        dispdata = append!(dispdata,string.(disp))
        peakdata =append!(peakdata,vec(max_infs')./Lc_target)

        min_I_vect = zeros(sim_ens)
        max_I_vect = zeros(sim_ens)
        bound_starts = zeros(sim_ens)

        ncs = Bool.(ones(length(max_infs)))

        for kk in 1:sim_ens

            R_cross_one = diff(sign.(Rewvect[:,kk].-1.0)).<0
            indxs = 1:length(R_cross_one)

            indx_ss = indxs[R_cross_one][1]
            bound_starts[kk] = indx_ss

            reduced_I = Ivect[indx_ss:ndays, kk]

            I_cross_targ = diff(sign.(reduced_I.-Lc_target)).<0
            indxs_I = 1:length(I_cross_targ)

            if length(indxs_I[I_cross_targ]) < 1
                max_I_vect[kk] = maximum(Ivect[:, kk])#NaN
                min_I_vect[kk] = minimum(Ivect[:, kk])#NaN
                ncs[kk] = 0

            else

                indx_ss_I = indxs_I[I_cross_targ][1]
                bound_starts[kk] = indx_ss + indx_ss_I + 1
                
                max_I_vect[kk] = maximum(Ivect[(indx_ss + indx_ss_I):ndays, kk])
                min_I_vect[kk] = minimum(Ivect[(indx_ss + indx_ss_I):ndays, kk])
            end

        end
        
        bound_start = minimum(bound_starts)
        env_size = max_I_vect - min_I_vect
        envdata = append!(envdata,vec(env_size')./Lc_target)
        ncdata = append!(ncdata,vec(ncs'))
        costdata = append!(costdata,vec(sum_daily_costs'))
        costs2 = append!(costs2,mean(vec(cost_daily')))
        overshoot_days = append!(overshoot_days,mean(vec(overshoots')))


        #plt.scatter(xs, max_infs, color="blue", alpha=0.01)
    end

end
##

cd("figs4/delay_optimal_fig6")

delv = delv[2:end]
del_dispv = del_dispv[2:end]

ts = Ns+LDs+SDs
ts = ts[2:end]

Ns = Ns[2:end]    
LDs = LDs[2:end]
SDs = SDs[2:end]
ivs = (LDs+SDs)./ts
total = (Ns+LDs+SDs)./ts

Ns = Ns./ts
LDs = LDs./ts
SDs = SDs./ts

costs = LDcost*LDs+SDcost*SDs

df2 = Dict("delay" => delv,
"α" => del_dispv,
"N" => Ns,
"LD" => LDs,
"SD" => SDs,
"total" => total,
"ivs" => ivs,
"costs" => costs,
"costs2" => costs2,
"overshoot" => overshoot_days,)

sns.set(style="white")
sns.barplot(x="delay",y="total", hue="α", data=df2, color="green")
sns.barplot(x="delay",y="ivs", hue="α", data=df2, color="purple")
sns.barplot(x="delay",y="LD", hue="α", data=df2, color="red")
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.xlabel("Mean reporting delay [days]")
plt.ylabel("Interventions")
plt.legend([], [], frameon=false)
plt.savefig("interventions.png", dpi=300)
plt.savefig("interventions.svg")
plt.show()

# sns.set(style="white")
# sns.scatterplot(x="delay",y="N", hue="α", data=df2, color="green")
# sns.scatterplot(x="delay",y="SD", hue="α", data=df2, color="purple")
# sns.scatterplot(x="delay",y="LD", hue="α", data=df2, color="red")
# #plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
# plt.xlabel("Mean reporting delay [days]")
# plt.ylabel("Interventions")
# plt.legend([], [], frameon=false)
# #plt.savefig("interventions_thr.png", dpi=300)
# plt.show()

#show avg costs
sns.set(style="white")
#sns.barplot(x="delay",y="costs2", hue="α", data=df2, color="purple")
sns.barplot(x="delay",y="costs", hue="α", data=df2, color="purple")
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.xlabel("Mean reporting delay [days]")
plt.ylabel("Mean intervention costs")
plt.ylim([0,0.06])
plt.legend([], [], frameon=false)
plt.savefig("interv_costs.png", dpi=300)
plt.savefig("interv_costs.svg")
plt.show()

xdatar = xdata[sim_ens+1:end]
peakdatar = peakdata[sim_ens+1:end]
dispdatar = dispdata[sim_ens+1:end]
envdatar = envdata[sim_ens+1:end]
ncdatar = ncdata[sim_ens+1:end]

datafr = Dict("delay" => xdata,
"max_infs" => peakdata,
"α" => dispdata,
"env" => envdata,
"ncs" => ncdata,
"costs" => costdata)
    

sns.catplot(x="delay",y="max_infs", hue="α", data=datafr, legend=true, s=2, marker="D", alpha=0.2, dodge=true)
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.yscale("log")
plt.xlabel("Mean reporting delay [days]")
plt.ylabel("Peak incidence/Target")
plt.ylim([1,10^1])
plt.savefig("peaks.svg")
plt.savefig("peaks.png", dpi=300)
plt.show()

sns.catplot(x="delay",y="env", hue="α", data=datafr, legend=true, s=2, marker="D", alpha=0.2, dodge=true)
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.yscale("log")
plt.xlabel("Mean reporting delay [days]")
plt.ylabel("Envelope size/Target")
plt.ylim([10^(-1),10^1])
plt.savefig("envs.svg", dpi=300)
plt.savefig("envs.png", dpi=300)
plt.show()

cd("../..")