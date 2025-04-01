using Plots, CSV, DataFrames, Statistics, Distributions, Arrow


algorithms = ["VI", "PAI", "FT", "HK", "KB", "RCPI", "WIN", "PPI"]
#linetypes = [:solid, :dash, :dot]
#= algorithms = ["PAI", "KM"]
legend_labels = ["PAI", "RCPI"] =#

results = copy(DataFrame(Arrow.Table("Paper/new test data/inv_fast.arrow")))
alg_results = groupby(results, :algorithm)

p = plot(title = "Algorithm Solution Quality vs Time", xlabel = "Time in Seconds", ylabel = "||v - v⋆||∞", yscale = :log, size = (1200,800))

for big_item ∈ alg_results
    sort!(big_item, :runtime)
    item = big_item[400,:]
    if item.algorithm ∈ algorithms
        plot_err = copy(item.errors)
        plot_err[end] = 1e-3
        plot!(item.times,plot_err,label = item.algorithm)
    end

end

hline!([1e-3], linestyle = :dash, color = :red, label = "Tolerance")

display(p)

#= time_stats = combine(groupby(results, [:state_number,:algorithm,:γ]), :time => mean, nrow, :time => std)

plot1 = plot(title = "γ = .9", xlabel = "Number of States", ylabel = "Solve Time (seconds)")
for (i,a) ∈ enumerate(algorithms)
    cur = time_stats[(time_stats.γ .== .9) .& (time_stats.algorithm .== a), [:time_mean, :time_std, :nrow, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_mean, label = legend_labels[i], yerror = cur.time_std.*quantile.(TDist.(float(cur.nrow)),.975)#= , ls=:auto =#)
end
plot2 = plot(title = "γ = .95", xlabel = "Number of States", ylabel = "Solve Time (seconds)")
for (i,a) ∈ enumerate(algorithms)
    cur = time_stats[(time_stats.γ .== .95) .& (time_stats.algorithm .== a), [:time_mean, :time_std, :nrow, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_mean, label = legend_labels[i], yerror = cur.time_std.*quantile.(TDist.(float(cur.nrow)),.975)#= , ls=:auto =#)
end
plot3 = plot(title = "γ = .99", xlabel = "Number of States", ylabel = "Solve Time (seconds)")
for (i,a) ∈ enumerate(algorithms)
    cur = time_stats[(time_stats.γ .== .99) .& (time_stats.algorithm .== a), [:time_mean, :time_std, :nrow, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_mean, label = legend_labels[i], yerror = cur.time_std.*quantile.(TDist.(float(cur.nrow)),.975)#= , ls=:auto =#)
end =#

#= median_times = combine(groupby(results, [:state_number,:algorithm,:γ]), :time => median)
plot4 = plot(title = "median, γ = .9")
for a ∈ algorithms
    cur = median_times[(time_stats.γ .== .9) .& (time_stats.algorithm .== a), [:time_median, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_median, label = a)
end
plot5 = plot(title = "median, γ = .95")
for a ∈ algorithms
    cur = median_times[(time_stats.γ .== .95) .& (time_stats.algorithm .== a), [:time_median, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_median, label = a)
end
plot6 = plot(title = "median, γ = .99")
for a ∈ algorithms
    cur = median_times[(time_stats.γ .== .99) .& (time_stats.algorithm .== a), [:time_median, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_median, label = a)
end

max_times = combine(groupby(results, [:state_number,:algorithm,:γ]), :time => maximum)
plot7 = plot(title = "max, γ = .9")
for a ∈ algorithms
    cur = max_times[(time_stats.γ .== .9) .& (time_stats.algorithm .== a), [:time_maximum, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_maximum, label = a)
end
plot8 = plot(title = "max, γ = .95")
for a ∈ algorithms
    cur = max_times[(time_stats.γ .== .95) .& (time_stats.algorithm .== a), [:time_maximum, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_maximum, label = a)
end
plot9 = plot(title = "max, γ = .99")
for a ∈ algorithms
    cur = max_times[(time_stats.γ .== .99) .& (time_stats.algorithm .== a), [:time_maximum, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_maximum, label = a)
end =#

#plot(plot1,plot2,plot3,#= plot4,plot5,plot6,plot7,plot8,plot9, =# layout = (1,3), size = (900,600), ylim = (0,2), legend=:topleft)
