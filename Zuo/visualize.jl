using Random, Plots
using SparseArrays
using Distributed 
using JSON, Lasso, LARS, LaTeXStrings, NPZ
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
solver_asp = ACEfit.ASP(; P = P, select = :final, tsvd = true, actMax = 100,  loglevel = 1)
asp_result = ACEfit.solve(solver_asp, Wt .* At, Wt .* yt, Wv .* Av, Wv .* yv);

asp_final = set_parameters!( deepcopy(model), 
                  Array(ACEfit.asp_select(asp_result, :final)[1]))

pot_asp = fast_evaluator(asp_final; aa_static = true)  

@info("---------- ASP test errors ----------")
err_fin = ACEpotentials.compute_errors(test, pot_asp)

######### Visualization ###########

nnll = ACEpotentials.Models.get_nnll_spec(model.model)

i1000 = ACEfit.asp_select(asp_result, :final)[1].nzind
i300 = ACEfit.asp_select(asp_result, (:bysize, 300))[1].nzind
i100 = ACEfit.asp_select(asp_result, (:bysize, 100))[1].nzind
i30 = ACEfit.asp_select(asp_result, (:bysize, 30))[1].nzind

filter2(_nnll) = filter(bb -> length(bb) == 2, _nnll)
nnll_30 = filter2(nnll[i30[i30 .<= length(nnll)]])
nnll_100 = filter2(nnll[i100[i100 .<= length(nnll)]])
nnll_300 = filter2(nnll[i300[i300 .<= length(nnll)]])
nnll_1000 = filter2(nnll[i1000[i1000 .<= length(nnll)]])

nnll_full_30 = filter2(nnll[29:29+30])
nnll_full_100 = filter2(nnll[29:29+100])
nnll_full_300 = filter2(nnll[29:29+300])
nnll_full_1000 = filter2(nnll[29:29+1000])


dark2_colors = palette(:Dark2)

function plot_nl!(asp_nls, ace_nls, fin_nls, fig_name, annotate=:Bool)
    nl1 = [act[1][:l] + act[1][:n] for act in asp_nls]
    nl2 = [act[2][:l] + act[2][:n] for act in asp_nls]

    fin_nl1 = [act[1][:l] + act[1][:n] for act in fin_nls]
    fin_nl2 = [act[2][:l] + act[2][:n] for act in fin_nls]

    ace_nl1 = [act[1][:l] + act[1][:n] for act in ace_nls]
    ace_nl2 = [act[2][:l] + act[2][:n] for act in ace_nls]

    max_limit = max(maximum(fin_nl1), maximum(fin_nl2))

    plot(xlabel=L"\mathrm{n_1 + l_1}", ylabel=L"\mathrm{n_2 + l_2}", legend=false, size=(400,400),grid=false, 
      xtickfont=font(8, "Times"), ytickfont=font(8, "Times"), 
      )

    grid_size = 1.0 

    for (x, y) in zip(fin_nl1, fin_nl2)
        plot!([x, x+grid_size, x+grid_size, x, x], [y, y, y+grid_size, y+grid_size, y], lw=2, color = :black)
    end

    for (x, y) in zip(fin_nl2, fin_nl1)
        plot!([x, x+grid_size, x+grid_size, x, x], [y, y, y+grid_size, y+grid_size, y], lw=2,color = :black)
    end

    xlims!(0, max_limit)
    ylims!(0, max_limit)

    for (x, y) in zip(ace_nl1, ace_nl2)
        plot!([x, x+grid_size, x+grid_size, x, x],
            [y, y, y+grid_size, y+grid_size, y],
            lw=2, seriestype=:shape, fillcolor=dark2_colors[8], linecolor=:black)
        plot!([y, y, y+grid_size, y+grid_size, y], [x, x+grid_size, x+grid_size, x, x],
        lw=2, seriestype=:shape, fillcolor=dark2_colors[8], linecolor=:black)
    end

    for (x, y) in zip(nl1, nl2)
        plot!([x, x+grid_size, x+grid_size, x, x],
            [y, y, y+grid_size, y+grid_size, y],
            lw=2, seriestype=:shape, fillcolor=dark2_colors[4], linecolor=:black)
        plot!([y, y, y+grid_size, y+grid_size, y], [x, x+grid_size, x+grid_size, x, x],
        lw=2, seriestype=:shape, fillcolor=dark2_colors[4], linecolor=:black)
    end

    if annotate == true
        plot!([max_limit - 4, max_limit - 3, max_limit - 3, max_limit - 4, max_limit - 4], 
        [max_limit - 1, max_limit - 1, max_limit, max_limit, max_limit - 1], 
        lw=2, seriestype=:shape, fillcolor=dark2_colors[8], linecolor=:black, label=nothing)

        annotate!(max_limit - 1.8, max_limit - 0.5, text("ACE", :black, 8, :bold, "Times"))

        plot!([max_limit - 4, max_limit - 3, max_limit - 3, max_limit - 4, max_limit - 4], 
        [max_limit - 3, max_limit - 3, max_limit - 2, max_limit - 2, max_limit - 3], 
        lw=2, seriestype=:shape, fillcolor=dark2_colors[4], linecolor=:black, label=nothing)
        annotate!(max_limit - 1.8, max_limit - 2.5, text("ASP", :black, 8, :bold, "Times"))
    end
    
    savefig("$fig_name.pdf")  
end



plot_nl!(nnll_30, nnll_full_30, nnll_full_1000, "30",false)
plot_nl!(nnll_100, nnll_full_100, nnll_full_1000, "100",false)
plot_nl!(nl_300, nnll_full_300, nnll_full_1000, "300",false)
plot_nl!(nl_1000, nnll_full_1000, nnll_full_1000, "1000",true)

################### 3D #####################

filter3(_nnll) = filter(bb -> length(bb) == 3, _nnll)


nnll_30 = filter3(nnll[i30[i30 .<= length(nnll)]])
nnll_100 = filter3(nnll[i100[i100 .<= length(nnll)]])
nnll_300 = filter3(nnll[i300[i300 .<= length(nnll)]])
nnll_1000 = filter3(nnll[i1000[i1000 .<= length(nnll)]])

nnll_full_30 = filter2(nnll[29:29+30])
nnll_full_100 = filter2(nnll[29:29+100])
nnll_full_300 = filter2(nnll[29:29+300])
nnll_full_1000 = filter2(nnll[29:29+1000])


snapshots = [nnll_30, nnll_100, nnll_300, nnll_1000]
full_snapshots = [nnll_full_30, nnll_full_100,  nnll_full_300, nnll_full_1000]

nl1 = [act[1][:l] + act[1][:n] for act in snapshots[i]]
nl2 = [act[2][:l] + act[2][:n] for act in snapshots[i]]
nl3 = [act[3][:l] + act[3][:n] for act in snapshots[i]]

full_nl1 = [act[1][:l] + act[1][:n] for act in full_snapshots[i]]
full_nl2 = [act[2][:l] + act[2][:n] for act in full_snapshots[i]]
full_nl3 = [act[3][:l] + act[3][:n] for act in full_snapshots[i]]

### We do the plotting in Python ###
npzwrite("full_nl14.npy", full_nl1)
npzwrite("full_nl24.npy", full_nl2)
npzwrite("full_nl34.npy", full_nl3)
npzwrite("nl14.npy", nl1)
npzwrite("nl24.npy", nl2)
npzwrite("nl3.npy", nl3)

