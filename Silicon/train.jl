using Distributed
addprocs(31, exeflags="--project=$(Base.active_project())")
using Random, Plots
using SparseArrays
using Distributed 
using JSON, Lasso, LARS
using ACEpotentials.Models: fast_evaluator


train_xyz = "gp_iter6_sparse9k.xml.xyz"
data = JuLIP.read_extxyz(train_xyz)
shuffled_data = shuffle(data)

n = length(shuffled_data)
train_size = Int(floor(0.7 * n))  # 70% for training
valid_size = Int(floor(0.15 * n))  # 15% for validation

# Split the data
train_data = shuffled_data[1:train_size]
valid_data = shuffled_data[train_size+1:train_size+valid_size]
test_data = shuffled_data[train_size+valid_size+1:end]

#setup weights and solver
weights = Dict(
    "default" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
    "liq" => Dict("E" => 10.0, "F" => 0.66, "V" => 0.25 ),
    "amorph" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1),
    "sp" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1),
    "bc8"=> Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1),
    "vacancy" => Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1),
    "interstitial" => Dict("E" => 50.0, "F" => 0.5 , "V" => 0.1),
    # new
    "dia" => Dict("E" => 60.0, "F" => 2.0 , "V" => 2.0 ),
)

model = ace1_model(elements = [:Si], order = 4, totaldegree = 23,
                Eref = [:Si => -158.54496821]) 

P = algebraic_smoothness_prior(model; p = 5)

energy_key = "dft_energy"
force_key = "dft_force"
virial_key = "dft_virial"

At, yt, Wt = ACEpotentials.assemble(train_data, model, energy_key = energy_key,
                force_key = force_key, virial_key = virial_key, weights = weights) 

Av, yv, Wv = ACEpotentials.assemble(valid_data, model, energy_key = energy_key, 
                force_key = force_key, virial_key = virial_key, weights = weights) 


######### ASP ###########
solver_asp = ACEfit.ASP(; P = P, select = :final, tsvd = true, actMax = 5000,  loglevel = 1)
asp_result = ACEfit.solve(solver_asp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv);

asp_final = set_parameters!( deepcopy(model), 
                  Array(ACEfit.asp_select(asp_result, :final)[1]))

pot_asp = fast_evaluator(asp_final; aa_static = true)  

@info("---------- ASP test errors ----------")
err_fin = ACEpotentials.compute_errors(test, pot_asp)

######### OMP ###########

solver_omp = ACEfit.OMP(; P = P, select = :final, tsvd = false, actMax = 5000, loglevel = 1)
omp_result = ACEfit.solve(solver_omp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv)

omp_final = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(omp_result, :final)[1])
pot_omp = fast_evaluator(omp_final; aa_static = false)  

@info("---------- OMP test errors ----------")
ACEpotentials.linear_errors(test, pot_omp)
