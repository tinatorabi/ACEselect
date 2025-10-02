
using Distributed
addprocs(31, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials, JuLIP, LinearAlgebra, ActiveSetPursuit, Random


train_xyz = "train_data.xyz"
data = JuLIP.read_extxyz(train_xyz)
testdata = JuLIP.read_extxyz("test_data.xyz")

shuffled_data = shuffle(data)
n = length(shuffled_data)
train_size = Int(floor(0.85 * n))  # 85% for training
valid_size = Int(floor(0.15 * n))  # 15% for validation
train_data = shuffled_data[1:train_size]
valid_data = shuffled_data[train_size+1:train_size+valid_size]

model = ace1_model(elements = [:H, :O], order = 4, totaldegree = 14,
                Eref = [:H => -187.6043857100553, :O => -93.80219285502734]) 

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
solver_asp = ACEfit.ASP(; P = P, select = :final, tsvd = true, actMax = 12000,  loglevel = 1)
asp_result = ACEfit.solve(solver_asp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv);

asp_final = set_parameters!( deepcopy(model), 
                  Array(ACEfit.asp_select(asp_result, :final)[1]))

pot_asp = fast_evaluator(asp_final; aa_static = true)  

@info("---------- ASP test errors ----------")
err_fin = ACEpotentials.compute_errors(test, pot_asp)

######### OMP ###########

solver_omp = ACEfit.OMP(; P = P, select = :final, tsvd = false, actMax = 12000, loglevel = 1)
omp_result = ACEfit.solve(solver_omp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv)

omp_final = set_parameters!( deepcopy(model), 
                  ACEfit.asp_select(omp_result, :final)[1])
pot_omp = fast_evaluator(omp_final; aa_static = false)  

@info("---------- OMP test errors ----------")
ACEpotentials.linear_errors(test, pot_omp)

############################

i6000 = ACEfit.asp_select(asp_result, :final)[1]
i9000 = ACEfit.asp_select(asp_result, (:bysize, 300))[1]
omp_6000 = set_parameters!( deepcopy(model), i6000)
pot_6000 = fast_evaluator(omp_6000; aa_static = false)  

omp_9000 = set_parameters!( deepcopy(model), i9000)
pot_9000 = fast_evaluator(omp_9000; aa_static = false)  

######## Save the potentials #########
# open("$sym.json", "w") do f
# 	 JSON.print(f, "params" => omp_9000.ps)
# end
