using Plots, CSV, DataFrames, Statistics, Distributions, Arrow

@recipe function f(::Type{Val{:samplemarkers}}, x, y, z; number = 10, maxlen = 1000)
    n = sum(x .≤ maxlen)
    step = Int(ceil(n/number))
    sx, sy = x[[1:step:n;n]], y[[1:step:n;n]]
    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markershape --> :auto
    x := sx
    y := sy
end


algorithms = ["RCPI","PAI","VI", "FT", "HK", #= "KB", =# "WIN", "PPI"]
alg_marks = [:circle, :rect, :diamond, :hexagon, :cross, :dtriangle, :star5]
alg_colors = Plots.palette(:auto)[1:7]
alg_marks_dict = Dict(algorithms .=> alg_marks)
alg_colors_dict = Dict(algorithms .=> alg_colors)

results = copy(DataFrame(Arrow.Table("Paper/new test data/grid_all.arrow")))
alg_results = groupby(results, :algorithm)

p1 = plot(title = "Small Problems", yscale = :log, xlim = (0,60))

for big_item ∈ alg_results
    sort!(big_item, :runtime)
    item = big_item[400,:]
    if item.algorithm ∈ algorithms
        plot_err = copy(item.errors)
        plot_time = copy(item.times)
        if plot_err[end] < 1e-3
            z = (plot_err[end]/plot_err[end-1])^(1/(plot_time[end]-plot_time[end-1]))
            plot_time[end] = log(z,1e-3/plot_err[end-1]) + plot_time[end-1]
            plot_err[end] = 1e-3
        end
        plot!(plot_time,plot_err,label = item.algorithm, seriescolor = alg_colors_dict[item.algorithm], markershape = alg_marks_dict[item.algorithm],markerstrokewidth = .5, legend = :topright, seriestype = :samplemarkers, number = 5, maxlen = 60)
    end

end

hline!([1e-3], linestyle = :dash, color = :red, label = "Tolerance")

results = copy(DataFrame(Arrow.Table("Paper/new test data/grid_fast.arrow")))
alg_results = groupby(results, :algorithm)

p2 = plot(title = "Large Problems", yscale = :log)

for big_item ∈ alg_results
    sort!(big_item, :runtime)
    item = big_item[400,:]
    if item.algorithm ∈ algorithms
        plot_err = copy(item.errors)
        plot_time = copy(item.times)
        if plot_err[end] < 1e-3
            z = (plot_err[end]/plot_err[end-1])^(1/(plot_time[end]-plot_time[end-1]))
            plot_time[end] = log(z,1e-3/plot_err[end-1]) + plot_time[end-1]
            plot_err[end] = 1e-3
        end
        plot!(plot_time,plot_err,label = item.algorithm, seriescolor = alg_colors_dict[item.algorithm], markershape = alg_marks_dict[item.algorithm], markerstrokewidth = .5, seriestype = :samplemarkers, legend=false)
    end

end

hline!([1e-3], linestyle = :dash, color = :red, label = "Tolerance")


p = plot(p1,p2, layout= (1,2),plot_title = "Inventory Solution Quality vs Time", xlabel = "Time in Seconds", ylabel = "||Tv - v||∞", size = (1200,800),left_margin=5Plots.mm)
print(colors_used)
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
