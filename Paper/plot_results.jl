using Plots, CSV, DataFrames, Statistics, Distributions, Arrow, LaTeXStrings,Latexify

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


algorithms = ["RCPI","PAI","VI", "FT", "HK", "KB", "WIN", "PPI"]
alg_marks = [:circle, :rect, :diamond, :hexagon, :cross, :dtriangle, :star5, :utriangle]
alg_colors = Plots.palette(:auto)[1:8]
alg_marks_dict = Dict(algorithms .=> alg_marks)
alg_colors_dict = Dict(algorithms .=> alg_colors)


#= pleg = plot(legend = :outertopright)
for alg ∈ algorithms
    plot!([1], label = alg, seriescolor = alg_colors_dict[alg], markershape = alg_marks_dict[alg],markerstrokewidth = .5)
end
plot!(foreground_color=:white, background_color=:white,
      xaxis=false, yaxis=false, framestyle=:none) =#

results = copy(DataFrame(Arrow.Table("Paper/data/games_large.arrow")))
alg_results = groupby(results, :algorithm)

xmax = 5
p = plot(yscale = :log, xlim = (0,xmax), xlabel = "Time (s)", ylabel = L"\psi_\infty(v)", size = (600,400))


for big_item ∈ alg_results[2:3]
    sort!(big_item, :runtime)
    item = big_item[300,:]
    if item.algorithm ∈ algorithms
        plot_err = copy(item.errors)
        plot_time = copy(item.times)
        if plot_err[end] < 1e-3
            z = (plot_err[end]/plot_err[end-1])^(1/(plot_time[end]-plot_time[end-1]))
            plot_time[end] = log(z,1e-3/plot_err[end-1]) + plot_time[end-1]
            plot_err[end] = 1e-3
        end
        plot!(plot_time,plot_err,label = (item.algorithm == "KB" ? "RCPI∞" : item.algorithm == "RCPI" ? "RCPI₀" : item.algorithm == "WIN" ? "WS" : item.algorithm), seriescolor = alg_colors_dict[item.algorithm], markershape = alg_marks_dict[item.algorithm],markerstrokewidth = .5, legend = :topright, seriestype = :samplemarkers, number = 5, maxlen = xmax)
    end

end

for big_item ∈ alg_results[vcat([1],4:5)]
    sort!(big_item, :runtime)
    item = big_item[300,:]
    if item.algorithm ∈ algorithms
        plot_err = copy(item.errors)
        plot_time = copy(item.times)
        if plot_err[end] < 1e-3
            z = (plot_err[end]/plot_err[end-1])^(1/(plot_time[end]-plot_time[end-1]))
            plot_time[end] = log(z,1e-3/plot_err[end-1]) + plot_time[end-1]
            plot_err[end] = 1e-3
        end
        plot!(plot_time,plot_err,label = (item.algorithm == "KB" ? "RCPI∞" : item.algorithm == "RCPI" ? "RCPI₀" : item.algorithm == "WIN" ? "WS" : item.algorithm), seriescolor = alg_colors_dict[item.algorithm], markershape = alg_marks_dict[item.algorithm],markerstrokewidth = .5, legend = :topright, seriestype = :samplemarkers, number = 5, maxlen = xmax)
    end

end

hline!([1e-3], linestyle = :dash, color = :red, label = "Tolerance")





