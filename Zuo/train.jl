using Random, Plots
using SparseArrays
using Distributed 
using JSON, Lasso, LARS
using ACEpotentials.Models: fast_evaluator

addprocs(20, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials, JuLIP, LinearAlgebra, ActiveSetPursuit, Random


sym = :Li

@info("Reading data for $(sym)")
train, test, _ = ACEpotentials.example_dataset("Zuo20_$sym")
Random.seed!(1)
shuffled_indices = shuffle(1:length(train))
train_indices = shuffled_indices[1:round(Int, 0.85 * length(train))]
val_indices = shuffled_indices[round(Int, 0.85 * length(train)) + 1:end]
train_set = train[train_indices]
val_set = train[val_indices]

@info("---------- fitting $(sym) ----------")
model = ace1_model(elements = [sym], order = 3, totaldegree = 25, )
P = algebraic_smoothness_prior(model; p = 4)
weights = Dict("default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0))

energy_key = "energy"
force_key = "force"
virial_key = "virial"

At, yt, Wt = ACEpotentials.assemble(train_set, model, energy_key = energy_key,
                force_key = force_key, virial_key = virial_key, weights = weights) 

Av, yv, Wv = ACEpotentials.assemble(val_set, model, energy_key = energy_key, 
                force_key = force_key, virial_key = virial_key, weights = weights) 


######### ASP ###########
solver_asp = ACEfit.ASP(; P = P, select = :final, tsvd = true, actMax = 1000,  loglevel = 1)
asp_result = ACEfit.solve(solver_asp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv);

asp_final = set_parameters!( deepcopy(model), 
                  Array(ACEfit.asp_select(asp_result, :final)[1]))

pot_asp = fast_evaluator(asp_final; aa_static = true)  

@info("---------- ASP test errors ----------")
err_fin = ACEpotentials.compute_errors(test, pot_asp)

######### OMP ###########

solver_omp = ACEfit.OMP(; P = P, select = :final, tsvd = false, actMax = 1000, loglevel = 1)
omp_result = ACEfit.solve(solver_omp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv)

omp_final = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(omp_result, :final)[1])
pot_omp = fast_evaluator(omp_final; aa_static = false)  

@info("---------- OMP test errors ----------")
ACEpotentials.linear_errors(test, pot_omp)

######## Save the potentials #########
# open("$sym.json", "w") do f
# 	 JSON.print(f, "params" => omp_final.ps)
# end

######### BLR ###########

solver_blr = ACEfit.BLR(;)
blr_result = ACEfit.solve(solver_blr, (Wt .* At)/P, Wt .* yt)

blr_final = set_parameters!( deepcopy(model), P\blr_result["C"])
pot_blr = fast_evaluator(blr_final; aa_static = false)  

@info("---------- BLR test errors ----------")
ACEpotentials.linear_errors(test, pot_blr)


######### RRQR ###########

solver_rrqr = ACEfit.RRQR(P=P)
rrqr_result = ACEfit.solve(solver_rrqr, Wt .* At, Wt .* yt)

rrqr_final = set_parameters!( deepcopy(model), rrqr_result["C"])
pot_rrqr = fast_evaluator(rrqr_final; aa_static = false)  

@info("---------- RRQR test errors ----------")
ACEpotentials.linear_errors(test, pot_rrqr)

######### ARD ###########

solver_ard = ACEfit.SKLEARN_ARD(500, 0.1, 100) #vary the third arg for different sparsity
results_ard = ACEfit.solve(solver_ard, (Wt .* At)/P, Wt .* yt)
nz = count(x -> abs(x) > 1e-8, results_ard["C"])

ard_final = set_parameters!( deepcopy(model), P\results_ard["C"])
pot_ard = fast_evaluator(ard_final; aa_static = false)  

@info("---------- ARD test errors ----------")
ACEpotentials.linear_errors(test, pot_ard)

############################################
# Lasso.jl and LARS
############################################

path= fit(LassoPath, (Wt .* At)/P, Wt .* yt, λminratio=1e-6,stopearly=false, cd_maxiter=100000,)

λ_max = maximum(abs.(((Wt .* At)/P)' * (Wt .* yt))) / size(At, 1)
λ_min = λ_max * 1e-10
λ_seq = exp.(range(log(λ_max), log(λ_min), length=100))
path = fit(LassoPath, (Wt .* At)/P, Wt .* yt;
                  cd_maxiter=50000, standardize=true, intercept = false,
                  λ=λ_seq)

coef_matrix = path.coefs
coefs = path.coefs 
# Count active coefficients in each column
active_counts = count(!iszero, coefs; dims=1) 
counts = vec(active_counts) 
idx = argmin(abs.(counts .- 300))  # index of closest to 100
coef_approx300 = coefs[:, idx]

println("Closest count: ", counts[idx])
println("λ: ", path.λ[idx])


lasso_final = set_parameters!( deepcopy(model), P\coef_approx300)
pot_lasso = fast_evaluator(lasso_final; aa_static = false)  
ACEpotentials.linear_errors(test,pot_lasso)


# MAEs = []
# cfs = []
# lam = []
# for k in 10:10:100
#     lasso_final = set_parameters!( deepcopy(model), P\coef_matrix[:,k])
#     pot_lasso = fast_evaluator(lasso_final; aa_static = false)  
#     ACEpotentials.linear_errors(test,pot_lasso)
#     push!(MAEs, m)
#     push!(cfs,length(findall(!iszero, Bk)))
#     push!(lam, λ_seq[k])
# end

############### LARS ##################

lrs = lars((Wt .* At)/P, Wt .* yt; method=:lasso, intercept=true, standardize=true, lambda2=0.0,
            use_gram=true, maxiter=typemax(Int), lambda_min=0.73, verbose=true)

coef_path = lrs.coefs   
coef_final = coef_path[:, end]   # coefficients at the last step
nnz_coefs = count(!iszero, coef_final)


lars_final = set_parameters!( deepcopy(model), P\coef_final)
pot_lars = fast_evaluator(lars_final; aa_static = false)  
ACEpotentials.linear_errors(test, pot_lars)



