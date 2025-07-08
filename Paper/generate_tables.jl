using Plots, CSV, DataFrames, Statistics, Distributions, Arrow, LaTeXStrings,Latexify


results1 = copy(DataFrame(Arrow.Table("Paper/new test data/inv_all.arrow")))
results2 = copy(DataFrame(Arrow.Table("Paper/new test data/inv_fast_table.arrow"))) 
#results[results.runtime .≥ 1000, :runtime] .= Inf

standard_error(x) = std(x)/sqrt(length(x))


temp1 = combine(groupby(combine(groupby(results1, [:inv_id, :γ, :state_number]), [:runtime, :algorithm] => (t,a) -> t./t[a .== "VI"],
                                                                                :algorithm => a->a), :algorithm_function), :runtime_algorithm_function => mean => :mean,
                                                                                                                           :runtime_algorithm_function => standard_error => :standard_error,
                                                                                                                           :runtime_algorithm_function => (t->quantile(t,0.25)) => :lower_quartile, 
                                                                                                                           :runtime_algorithm_function => median => :median,
                                                                                                                           :runtime_algorithm_function => (t->quantile(t,0.75)) => :upper_quartile)
temp2 = combine(groupby(combine(groupby(results2, [:inv_id, :γ, :state_number]), [:runtime, :algorithm] => (t,a) -> t./t[a .== "VI"],
                                                                                :algorithm => a->a), :algorithm_function), :runtime_algorithm_function => mean => :big_mean)

temp = leftjoin(temp1,temp2,on=:algorithm_function)
                                                                                                                           
show(temp,allrows = true)
println()
copy_to_clipboard(true)
latexify(temp; env = :tabular, booktabs = true, fmt = "%.6f", latex = false)
