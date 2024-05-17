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

ur_dispersion = [2.0, 8.0, 20.0, 50.0]
urs = reverse([0.1, 0.25, 0.4, 0.55, 0.7, 0.85])

##Plot peak incidence

ur_disp = delay_dispersion[1]
ur = urs[1]
del = delays[6]

LDcost = 0.15
SDcost = 0.01

rf = 7
γ = 0.95
ndays = 41*7 #epidemic length
Lc_target = 5000 #desired infectiousness

#filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur).jld"
#filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_noS_cyclic_5_2.jld"
#filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_noS.jld"
#filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_noS_thr.jld"
#filename = "rerun2_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_noS_covid.jld"
#filename = "rerun2_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_noS_thr_lambda_covid.jld"
#filename = "rerun2_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_noS_cyclic_5_2.jld"

filename = "rerun3_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_noS_BIN_ebola.jld"

#filename = "rerun2_rf_3_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_threshold_then_cyclic_ur_21_36_BIN_ebola.jld"

#filename = "rerun3_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(ur)_noS_thr_inc_ebola_BIN.jld"
data = load(filename)

mean_of_peaks = zeros(length(delays))

Ivect = data["Ivect"]
Rewvect = data["Rewvect"]
max_infs = maximum(Ivect[1:ndays,:], dims=1)'
xs = ur*ones(length(max_infs))
disp = ur_dispersion[1]*ones(length(max_infs))
mean_of_peaks[1] = mean(max_infs)
all_infs = sum(Ivect[1:ndays,:], dims=1)'

sim_ens=size(Ivect[1:ndays,:])[2]

policy = data["policy"]
cost_daily = zeros(size(policy[1:ndays,:]))
cost_daily[policy[1:ndays,:].==3] .= LDcost
cost_daily[policy[1:ndays,:].==2] .= SDcost

#cost_daily[Ivect.>1.5*Lc_target] = cost_daily[Ivect.>1.5*Lc_target] .+ 5.0
sum_daily_costs = 1/ndays*sum(cost_daily, dims=1)'
#sum_daily_costs = sum(cost_daily, dims=1)'

Ns = [sum(policy[1:ndays,:].==1)]
SDs = [sum(policy[1:ndays,:].==2)]
LDs = [sum(policy[1:ndays,:].==3)]

xdata = []#string.(xs)
dispdata = []#string.(disp)
peakdata = []#vec(max_infs')
sumdata = []#vec(all_infs')
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

    if length(indxs_I[I_cross_targ]) < 1
        max_I_vect[kk] = NaN
        min_I_vect[kk] = NaN
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

ur2 = []
a2 = []

for ii in 6:6
    for jj in [1,2,3,4]
        for mm in 1:length(urs)

            ur2 = append!(ur2, urs[mm])
            a2 = append!(a2, ur_dispersion[jj])

            ur = urs[mm]
            ur_disp = jj

            #filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur).jld"
            #filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(ur_disp)_under_$(ur)_noS.jld"
            #filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(ur_disp)_under_$(ur)_noS_thr.jld"
            #filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(1)_under_$(0.7)_noS_ebola.jld"
            #filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(jj)_under_$(ur)_noS_thr_lambda.jld"
            #filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(jj)_under_$(ur)_noS_cyclic_5_2.jld"
            #filename = "rerun2_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(ur_disp)_under_$(ur)_noS_covid.jld"
            #filename = "rerun2_rf_$(rf)_gamma_$(γ)_no_del_urv_$(ur_disp)_under_$(ur)_noS_thr_lambda_covid.jld"
            #filename = "rerun2_rf_$(rf)_gamma_$(γ)_no_del_urv_$(ur_disp)_under_$(ur)_noS_cyclic_5_2.jld"

            filename = "rerun3_opt_rf_$(rf)_gamma_$(γ)_no_del_urv_$(ur_disp)_under_$(ur)_noS_BIN_ebola.jld"
            
            #filename = "rerun2_rf_3_gamma_$(γ)_no_del_urv_$(ur_disp)_under_$(ur)_threshold_then_cyclic_ur_21_36_BIN_ebola.jld"

            #filename = "rerun3_rf_$(rf)_gamma_$(γ)_no_del_urv_$(ur_disp)_under_$(ur)_noS_thr_inc_ebola_BIN.jld"
            data = load(filename)

            #Plot peak incidence

            Ivect = data["Ivect"]
            Rewvect = data["Rewvect"]
            max_infs = maximum(Ivect[1:ndays,:], dims=1)'
            xs = ur*ones(length(max_infs))
            disp = ur_dispersion[jj]*ones(length(max_infs))

            policy = data["policy"]

            cost_daily = zeros(size(policy[1:ndays,:]))

            cost_daily[policy[1:ndays,:].==3] .= LDcost
            cost_daily[policy[1:ndays,:].==2] .= SDcost

            #cost_daily[Ivect.>1.5*Lc_target] = cost_daily[Ivect.>1.5*Lc_target] .+ 5.0

            sum_daily_costs = 1/ndays*sum(cost_daily, dims=1)'

            Ns = append!(Ns,[sum(policy[1:ndays,:].==1)])
            SDs = append!(SDs,[sum(policy[1:ndays,:].==2)])
            LDs = append!(LDs,[sum(policy[1:ndays,:].==3)])

            mean_of_peaks[ii] = mean(max_infs)
            all_infs = sum(Ivect[1:ndays,:], dims=1)'

            xdata = append!(xdata,string.(xs))
            dispdata = append!(dispdata,string.(disp))
            peakdata =append!(peakdata,vec(max_infs')./(Lc_target/ur))
            sumdata = append!(sumdata,vec(all_infs'))

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

                I_cross_targ = diff(sign.(reduced_I.-(Lc_target/ur))).<0
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
            envdata = append!(envdata,vec(env_size')./(Lc_target/ur))
            ncdata = append!(ncdata,vec(ncs'))
            costdata = append!(costdata,vec(sum_daily_costs'))

        end


        #plt.scatter(xs, max_infs, color="blue", alpha=0.01)
    end

end

xdatar = xdata#[sim_ens+1:end]
peakdatar = peakdata#[sim_ens+1:end]
dispdatar = dispdata#[sim_ens+1:end]
ncdatar = ncdata#[sim_ens+1:end]
envdatar = envdata#[sim_ens+1:end]
sumdatar = sumdata#[sim_ens+1:end]

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

datafr = Dict("ur" => xdatar,
"max_infs" => peakdatar,
"a" => dispdatar,
"env" => envdatar,
"ncs" => ncdatar,
"all_infs" => sumdatar,
"costs" => costdata,
"N" => Ns,
"LD" => LDs,
"SD" => SDs,
"total" => total,
"ivs" => ivs,
"costs2" => costs,
"ur2" => ur2,
"a2" => a2)
    
cd("figs4/ur_optimal_fig7")

# plt.hist(overshoots)
# plt.xlabel("Overshoot")
# plt.show()

#sns.distplot( a=max_infs)
#plt.xlabel("Peak incidence")
#plt.plot(delays, mean_of_peaks, color="black", marker="D")
#plt.xlabel("Mean reporting delay [days]")
#plt.ylabel("Peak incidence")
#plt.show()

# plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
# sns.swarmplot(x="delay",y="max_infs", hue="delay", data=datafr, legend=false, s=0.8, marker="D", alpha=1)
# plt.show()


sns.catplot(x="ur",y="max_infs", hue="a", data=datafr, legend=true, s=2, marker="D", alpha=0.2, dodge=true)
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.yscale("log")
plt.xlabel("Mean reportng rate")
plt.ylabel("Peak incidence/Target")
plt.ylim([0.1,10^1])
plt.savefig("peaks_ur.png", dpi=300)
plt.savefig("peaks_ur.svg", dpi=300)
plt.show()

sns.catplot(x="ur",y="env", hue="a", data=datafr, legend=true, s=2, marker="D", alpha=0.2, dodge=true)
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.yscale("log")
plt.xlabel("Mean reportng rate")
plt.ylabel("Envelope size/Target")
plt.ylim([0.1,10^1])
plt.savefig("env_ur.png", dpi=300)
plt.savefig("env_ur.svg", dpi=300)
plt.show()


sns.barplot(x="ur",y="costs", hue="a", data=datafr, color="purple", errorbar=nothing)
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.xlabel("Mean reporting rate")
plt.ylabel("Mean intervention costs")
plt.ylim([0,0.06])
plt.legend([], [], frameon=false)
plt.savefig("costs_ur2.png", dpi=300)
plt.savefig("costs_ur2.svg", dpi=300)
plt.show()


sns.barplot(x="ur2",y="total", hue="a2", data=datafr, color="green")
sns.barplot(x="ur2",y="ivs", hue="a2", data=datafr, color="purple")
sns.barplot(x="ur2",y="LD", hue="a2", data=datafr, color="red")
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.xlabel("Mean reporting rate")
plt.ylabel("Interventions")
plt.legend([], [], frameon=false)
plt.savefig("interventions.png", dpi=300)
plt.savefig("interventions.svg", dpi=300)
plt.show()

cd("../..")