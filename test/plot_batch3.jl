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
delay_dispersion =[0.1, 1.0, 5.0, 25.0, 200.0]
under_reporting = [1.0]

##Plot peak incidence

del_disp = delay_dispersion[2]
ur = under_reporting[1]
del = delays[1]

rf = 7
γ = 0.95
ndays = 21*7 #epidemic length
Lc_target = 5000 #desired infectiousness

#filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur).jld"
#filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS.jld"
#filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_ebola.jld"
#filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold.jld"
filename = "rerun3_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_covid.jld"
data = load(filename)

mean_of_peaks = zeros(length(delays))

Ivect = data["Ivect"]
Rewvect = data["Rewvect"]
policy = data["policy"]

Ns = [sum(policy.==1)]
SDs = [sum(policy.==2)]
LDs = [sum(policy.==3)]

max_infs = maximum(Ivect[1:ndays,:], dims=1)'
xs = del*ones(length(max_infs))
disp = del_disp*ones(length(max_infs))
mean_of_peaks[1] = mean(max_infs)
all_infs = sum(Ivect[1:ndays,:], dims=1)'

sim_ens=size(Ivect)[2]

xdata = string.(xs)
dispdata = string.(disp)
peakdata = vec(max_infs')
sumdata = vec(all_infs')

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
envdata = vec(env_size')
ncdata = vec(ncs')

for ii in 1:6
    for jj in [2,3,4,5]

        del = delays[ii]
        del_disp = delay_dispersion[jj]

        #filename = "workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur).jld"
        #filename = "longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_ebola.jld"
        #filename = "workspace_variables_opt_del_$(del)_delv_$(del_disp)_under_$(ur)_threshold.jld"
        filename = "rerun3_longer_workspace_variables_opt_rf_$(rf)_gamma_$(γ)_del_$(del)_delv_$(del_disp)_under_$(ur)_noS_covid.jld"
        data = load(filename)

        #Plot peak incidence

        Ivect = data["Ivect"]
        Rewvect = data["Rewvect"]

        policy = data["policy"]

        Ns = append!(Ns,[sum(policy.==1)])
        SDs = append!(SDs,[sum(policy.==2)])
        LDs = append!(LDs,[sum(policy.==3)])

        max_infs = maximum(Ivect[1:ndays,:], dims=1)'
        xs = del*ones(length(max_infs))
        disp = del_disp*ones(length(max_infs))

        mean_of_peaks[ii] = mean(max_infs)
        all_infs = sum(Ivect[1:ndays,:], dims=1)'

        xdata = append!(xdata,string.(xs))
        dispdata = append!(dispdata,string.(disp))
        peakdata =append!(peakdata,vec(max_infs'))
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
        envdata = append!(envdata,vec(env_size'))
        ncdata = append!(ncdata,vec(ncs'))


        #plt.scatter(xs, max_infs, color="blue", alpha=0.01)
    end

end

target = 5000

xdatar = xdata[sim_ens+1:end]
peakdatar = peakdata[sim_ens+1:end]./target
dispdatar = dispdata[sim_ens+1:end]
envdatar = envdata[sim_ens+1:end]./target
ncdatar = ncdata[sim_ens+1:end]
sumdatar = sumdata[sim_ens+1:end]

datafr = Dict("delay" => xdatar,
"max_infs" => peakdatar,
"α" => dispdatar,
"env" => envdatar,
"ncs" => ncdatar,
"all_infs" => sumdatar)
    

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


sns.catplot(x="delay",y="max_infs", hue="α", data=datafr, legend=true, s=2, marker="D", alpha=0.2, dodge=true)
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.yscale("log")
plt.xlabel("Mean reporting delay [days]")
plt.ylabel("Peak incidence")
plt.show()

sns.catplot(x="delay",y="env", hue="α", data=datafr, legend=true, s=2, marker="D", alpha=0.2, dodge=true)
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.yscale("log")
plt.xlabel("Mean reporting delay [days]")
plt.ylabel("Envelope size")
plt.show()

sns.countplot(x="ncs", hue="delay", data=datafr)
#plt.plot(0:length(delays)-1, mean_of_peaks, color="black", marker="D")
plt.xlabel("Mean reporting delay [days]")
plt.ylabel("Envelope size")
plt.show()

sns.set(style="white")
sns.displot(data=datafr, x="max_infs", shrink=0.9, hue="α", kde=true, fill=true, alpha=0.2, bins=40)
plt.xlabel("Peak incidence")
#plt.savefig("peaks_hists_delay_($del).png", dpi=300)
plt.show()

sns.displot(data=datafr, x="env", shrink=0.9, hue="α", kde=true, fill=true, alpha=0.2, bins=40)
plt.xlabel("Envelope size")
#plt.savefig("envs_hists_delay_($del).png", dpi=300)
plt.show()

sns.displot(data=datafr, x="all_infs", shrink=0.9, hue="α", kde=true, fill=true, alpha=0.2, bins=40)
plt.xlabel("Total incidence")
#plt.savefig("Tot_inc_hists_delay_($del).png", dpi=300)
plt.show()