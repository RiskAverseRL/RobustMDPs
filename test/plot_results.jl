using Plots, CSV, DataFrames, Statistics


algorithms = ["VI", "PAI", "HK", "M1", "K1","KM", "WIN", "PPI"]
algorithms = ["PAI", "K1", "KM"]

results = DataFrame(CSV.File("Inventory_fast.csv"))

mean_times = combine(groupby(results, [:state_number,:algorithm,:γ]), :time => mean)
plot1 = plot(title = "mean, γ = .9")
for a ∈ algorithms
    cur = mean_times[(mean_times.γ .== .9) .& (mean_times.algorithm .== a), [:time_mean, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_mean, label = a)
end
plot2 = plot(title = "mean, γ = .95")
for a ∈ algorithms
    cur = mean_times[(mean_times.γ .== .95) .& (mean_times.algorithm .== a), [:time_mean, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_mean, label = a)
end
plot3 = plot(title = "mean, γ = .99")
for a ∈ algorithms
    cur = mean_times[(mean_times.γ .== .99) .& (mean_times.algorithm .== a), [:time_mean, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_mean, label = a)
end

median_times = combine(groupby(results, [:state_number,:algorithm,:γ]), :time => median)
plot4 = plot(title = "median, γ = .9")
for a ∈ algorithms
    cur = median_times[(mean_times.γ .== .9) .& (mean_times.algorithm .== a), [:time_median, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_median, label = a)
end
plot5 = plot(title = "median, γ = .95")
for a ∈ algorithms
    cur = median_times[(mean_times.γ .== .95) .& (mean_times.algorithm .== a), [:time_median, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_median, label = a)
end
plot6 = plot(title = "median, γ = .99")
for a ∈ algorithms
    cur = median_times[(mean_times.γ .== .99) .& (mean_times.algorithm .== a), [:time_median, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_median, label = a)
end

max_times = combine(groupby(results, [:state_number,:algorithm,:γ]), :time => maximum)
plot7 = plot(title = "max, γ = .9")
for a ∈ algorithms
    cur = max_times[(mean_times.γ .== .9) .& (mean_times.algorithm .== a), [:time_maximum, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_maximum, label = a)
end
plot8 = plot(title = "max, γ = .95")
for a ∈ algorithms
    cur = max_times[(mean_times.γ .== .95) .& (mean_times.algorithm .== a), [:time_maximum, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_maximum, label = a)
end
plot9 = plot(title = "max, γ = .99")
for a ∈ algorithms
    cur = max_times[(mean_times.γ .== .99) .& (mean_times.algorithm .== a), [:time_maximum, :state_number]]
    sort!(cur, [:state_number])
    plot!(cur.state_number,cur.time_maximum, label = a)
end

plot(plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,plot9, layout = (3,3), size = (1600,900), ylim = (0,200), legend=:topleft) #, xlabel= "Number of States", ylabel = "runtime in seconds")
