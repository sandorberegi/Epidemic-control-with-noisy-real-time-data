module EpiCont

    #Call other pkgs.
    using Random, Distributions
    using Base.Threads
    using PyCall, Conda

    Conda.add("seaborn")
    plt = pyimport("matplotlib.pyplot")
    np = pyimport("numpy")
    sns = pyimport("seaborn")

    export init_policies
    export reward_bilin
    export reward_quad
    export reward_sel
    export EpiRun_preds
    export EpiPred_till_wind_preds
    
    function init_policies(ctrl_states,dec)
        nstates = length(ctrl_states)
        policies = (nstates)

        if dec >= 2
            policies = (policies,policies)

            for ii in 3:dec
                policies = (policies..., length(ctrl_states))
            end
        end

        policies_tpl = []

        for ii in 0:nstates^dec-1
            sel_arr = zeros(Int64, dec)
            for jj in 1:dec
                sel_arr[jj] = mod(Int(floor(ii//nstates^(dec-jj))),nstates)+1
            end
            push!(policies_tpl,sel_arr)
        end

        return nstates, policies, policies_tpl
    end

    function reward_bilin(Lc, Re, Lc_pred, Re_pred, Lc_target, Lc_target_pen, R_target, jj, alpha, beta, ovp, cost_of_state)
              
        Lc_err_pred = abs(Lc_pred - Lc_target)
        R_err_pred = abs(Re_pred - R_target)
        over_pen = 0.0
        if(Lc_pred) > Lc_target_pen
            over_pen = ovp
        end
        return alpha * (-Lc_err_pred) + beta * (- R_err_pred) - cost_of_state[jj] - over_pen
    end

    function reward_quad(Lc, Re, Lc_pred, Re_pred, Lc_target, Lc_target_pen, R_target, jj, alpha, beta, ovp, cost_of_state)
        Lc_err_pred = (Lc_pred - Lc_target)
        R_err_pred = (Re_pred - R_target)

        if Lc_err_pred > 0.0
            Lc_err_quad = Lc_err_pred.^2
        else
            Lc_err_quad = 0.0
        end

        if R_err_pred > 0.0
            R_err_quad = R_err_pred.^2
        else
            R_err_quad = 0.0
        end
        
        over_pen = 0.0
        if(Lc_pred) > Lc_target_pen
            over_pen = ovp
        end
        return alpha * (-Lc_err_quad) + beta * (- R_err_quad) - cost_of_state[jj]# - over_pen
    end

    function reward_sel(cost_sel)
        if cost_sel == 1
            return reward_bilin
        elseif cost_sel == 2
            return reward_quad
        else
            println("This selection is not available. Using bilinear cost function.")
            return reward_bilin
        end
    end

    # #Running the epidemic with multi-step prediction
    function EpiRun_preds(Ivect, Revect, Lvect, Lcvect, Ldvect, Dvect, Svect, cvect, Rewvect, policy, Rest, R0est, w, wd, reward_fun, par)
        use_S = par.use_S
        N = par.N
        R_est_wind = par.R_est_wind
        I_min = par.I_min
        R0 = par.R0
        policies = par.policies
        δ = par.δ
        ρ = par.ρ
        R_coeff = par.R_coeff
        rf = par.rf
        n_ens = par.n_ens
        nYdel = par.nYdel
        delay_calc_v = par.delay_calc_v
        distr_sel = par.distr_sel
        binn = par.binn
        under_rep_calc = par.under_rep_calc

        ρcalc = 1.0
        if under_rep_calc == 1
            ρcalc = ρ
        end
        
        for ii in 2:length(Ivect)

            R_coeff_tmp = 0.0
            if use_S == 1
                if ii-1 < R_est_wind
                    Rest[ii] = mean(cvect[1:ii-1])/mean(Lcvect[1:ii-1])/mean(Svect[1:ii-1]).*N
                    R_coeff_tmp = sum(w[1:ii-1].* R_coeff[policy[ii-1:-1:1]])/sum(w[1:ii-1])
                    #R_coeff_tmp = mean(R_coeff[policy[1:ii-1]])
                else
                    Rest[ii] = mean(cvect[ii-R_est_wind:ii-1])/mean(Lcvect[ii-R_est_wind:ii-1])/mean(Svect[ii-R_est_wind:ii-1]).*N
                    #R_coeff_tmp = sum(w[1:ii-1].* R_coeff[policy[ii-1:-1:1]])/sum(w[1:ii-1])
                    R_coeff_tmp = sum(w[1:ii-R_est_wind].* R_coeff[policy[ii-R_est_wind:-1:1]])/sum(w[1:ii-R_est_wind])
                    #R_coeff_tmp = mean(R_coeff[policy[ii-R_est_wind:ii-1]])
                end
            else
                if ii-1 < R_est_wind
                    Rest[ii] = mean(cvect[1:ii-1])/mean(Lcvect[1:ii-1])
                    R_coeff_tmp = sum(w[1:ii-1].* R_coeff[policy[ii-1:-1:1]])/sum(w[1:ii-1])
                    #R_coeff_tmp = mean(R_coeff[policy[1:ii-1]])
                else
                    Rest[ii] = mean(cvect[ii-R_est_wind:ii-1])/mean(Lcvect[ii-R_est_wind:ii-1])
                    #R_coeff_tmp = sum(w[1:ii-1].* R_coeff[policy[ii-1:-1:1]])/sum(w[1:ii-1])
                    R_coeff_tmp = sum(w[1:ii-R_est_wind].* R_coeff[policy[ii-R_est_wind:-1:1]])/sum(w[1:ii-R_est_wind])
                    #R_coeff_tmp = mean(R_coeff[policy[ii-R_est_wind:ii-1]])
                end
            end
            
            #R0est[ii] = Rest[ii]/R_coeff[policy[ii-1]]
            R0est[ii] = Rest[ii]/R_coeff_tmp

            if Ivect[ii-1] < I_min
                policy[ii] = 1
                R0act = R0*R_coeff[policy[ii]]
            #elseif ii<= repd
            #    policy[ii] = 1
            #    R0act = R0*R_coeff[policy[ii]]
            else

                if mod(ii,rf) == 0
                    Rewards = zeros(policies) #intialise rewards
                    # predict the effect of the different strategies
                    for jj in 1:length(Rewards)
                        Rewards_ens = zeros(n_ens)
                        for kk in 1:n_ens
                            Rewards_ens[kk] = EpiPred_till_wind_preds(cvect[1:end], Revect[1:end], Lcvect[1:end], Svect[1:end], R0est[1:end], cvect[1:end], Lcvect[1:end], w, ii, jj, reward_fun, par) #- cost_of_state[jj]
                        end
                        reward = mean(Rewards_ens)
                        Rewards[jj] = reward
                    end

                    #println(Rewards)

                    #println(argmax(Rewards))

                    policy[ii] = argmax(Rewards)[1]
                    #R0act = R0*R_coeff[argmax(Rewards)]
                else
                    policy[ii] = policy[ii-1]
                end
                
                R0act = R0*R_coeff[policy[ii]]
            end
            Revect[ii] = R0act*Svect[ii-1]/N
            Rewvect[ii] = sum(w[1:ii-1].* Revect[ii:-1:2])/sum(w[1:ii-1])
            Lvect[ii] = sum(Ivect[ii-1:-1:1].* w[1:ii-1])
            Pois_input = sum(Ivect[ii-1:-1:1].* w[1:ii-1].* Revect[ii:-1:2])
            #if Lvect[ii]<Lvect[ii]*Revect[ii]
            #    Ivect[ii] = 0
            #else

            if Pois_input < 0
                Ivect[ii] = 0
            elseif distr_sel == 1
                X = Poisson(Pois_input)    
                Ivect[ii] = rand(X,1)[1]
            elseif distr_sel == 2
                X = Binomial(Int(round(Pois_input*binn)), 1/binn)
                Ivect[ii] = rand(X,1)[1]
            else
                Ivect[ii] = Int(round(Pois_input))
            end

            Ldvect[ii] = sum(Ivect[ii-1:-1:1].* wd[1:ii-1])
            Xd = Poisson(Ldvect[ii]*δ)    
            Dvect[ii] = rand(Xd,1)[1]
            #X = Poisson(Pois_input)    
            #Ivect[ii] = rand(X,1)[1]
            #end
            
            if delay_calc_v == 1
                Pois_input_c = sum(ρcalc.*Ivect[ii-1:-1:1].* w[1:ii-1].* nYdel[1:ii-1].* Revect[ii:-1:2])./sum(w[1:ii-1].* nYdel[1:ii-1])
            else
                Pois_input_c = sum(ρcalc.*Ivect[ii-1:-1:1].* nYdel[1:ii-1])./sum(nYdel[1:ii-1])
            end
            if Pois_input_c < 0
                cvect[ii] = 0
            elseif distr_sel == 1
                Xc = Poisson(Pois_input_c)    
                cvect[ii] = rand(Xc,1)[1]
            elseif distr_sel == 2
                Xc = Binomial(Int(round(Pois_input_c*binn)), 1/binn)
                cvect[ii] = rand(Xc,1)[1]
            else
                cvect[ii] = Int(round(Pois_input_c))
            end

            if under_rep_calc == 0
                Xcc = Binomial(Int(round(ρ*cvect[ii]*binn)), 1/binn)
                cvect[ii] = rand(Xcc,1)[1]
            end

            #cvect[ii] = trunc(Int, (Ivect[ii]*ρ))
            Lcvect[ii] = sum(cvect[ii-1:-1:1].* w[1:ii-1])
            Svect[ii] = Svect[ii-1]-Ivect[ii]
            if Svect[ii] < 0
                Svect[ii] = 0
            end
            #Ivect[ii] = Ivect[ii]-Dvect[ii] #the dead stop becoming infectious *(but in reality this does not decrease the number of new infections!)
            #println(Svect[ii])

        end
        return Ivect, Revect, Lvect, Lcvect, Dvect, Svect, cvect, Rewvect, policy, Rest, R0est
    end

    function EpiPred_till_wind_preds(Ivect, Revect, Lvect, Svect, R0est, cvect, Lcvect, w, kk, jj, reward_fun, par)

        pred_days = par.pred_days
        ndays = par.ndays
        rf = par.rf
        dec = par.dec
        alpha = par.alpha
        beta = par.beta
        ovp = par.ovp
        γ = par.γ
        R_coeff = par.R_coeff
        policies_tpl = par.policies_tpl
        use_S = par.use_S
        ρ = par.ρ
        Lc_target = par.Lc_target
        Lc_target_pen = par.Lc_target_pen
        R_target = par.R_target
        cost_of_state = par.cost_of_state
        use_inc = par.use_inc
        distr_sel = par.distr_sel
        binn = par.binn

        pred_window_end = kk+pred_days
    
        if pred_window_end > length(Ivect)
            pred_window_end = length(Ivect)
        end
    
        #println(kk)
        #println(Svect[kk-1:pred_window_end])
        
        for ii in kk:pred_window_end
            get_ridx = Int(floor((ii-kk)/rf))+1
            if get_ridx>dec
                get_ridx=dec
            end
            Ridx = policies_tpl[jj][get_ridx]
            R0act = R0est[kk]*R_coeff[Ridx]
            #println(R0act)
            #println(Ridx)
            if use_S == 1
                Revect[ii] = R0act*Svect[ii-1]/N
            else
                Revect[ii] = R0act#*Svect[ii-1]/N
            end
            Lvect[ii] = sum(cvect[ii-1:-1:1].* w[1:ii-1])
            Pois_input = sum(cvect[ii-1:-1:1].* w[1:ii-1].* Revect[ii:-1:2])
            #if Lvect[ii]<Lvect[ii]*Revect[ii]
            #    Ivect[ii] = 0
            #else
            #println(ii)
            if Pois_input > 0
                if distr_sel == 1
                    X = Poisson(Pois_input)    
                    Ivect[ii] = rand(X,1)[1]
                elseif distr_sel == 2
                    X = Binomial(Int(round(Pois_input*binn)), 1/binn)
                    Ivect[ii] = rand(X,1)[1]
                else
                    Ivect[ii] = Int(round(Pois_input))
                end
            else
                Ivect[ii] = 0
            end
          
            cvect[ii] = Ivect[ii]
            Lcvect[ii] = sum(cvect[ii-1:-1:1].* w[1:ii-1])
            Svect[ii] = Svect[ii-1]-Ivect[ii]
    
        end
    
        rew = zeros(ndays+pred_days)
        discounts = ones(ndays+pred_days)
    
        #println(R0act)
        #println(Svect[kk+1:pred_window_end])
        #println(Revect[kk+1:pred_window_end])
        #println(Ivect[kk+1:pred_window_end])
    
        for ii in kk+1:pred_window_end
            discounts[ii] = discounts[ii-1]*γ
            Ic = Ivect[ii-1]
            Lc = Lvect[ii-1]
            Re = Revect[ii-1]
            Lc_pred = Lvect[ii]
            Ic_pred = Ivect[ii]
            Re_pred = Revect[ii]
    
            get_ridx = Int(floor((ii-kk)/rf))+1
            if get_ridx>dec
                get_ridx=dec
            end
            Ridx = policies_tpl[jj][get_ridx]
            
            if use_inc == 1
                rew[ii] = reward_fun(Ic, Re, Ic_pred, Re_pred, Lc_target, Lc_target_pen, R_target, Ridx, alpha, beta, ovp, cost_of_state)
            else
                rew[ii] = reward_fun(Lc, Re, Lc_pred, Re_pred, Lc_target, Lc_target_pen, R_target, Ridx, alpha, beta, ovp, cost_of_state)
            end
    
        end
    
        return sum(rew.*discounts)
    end

end
