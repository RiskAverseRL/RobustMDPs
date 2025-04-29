include("game_algorithm_definition.jl")

using Plots,Serialization,LaTeXStrings
# Used to compute Ψ(v) for v's in span of (1,0,0), (0,-1,1) in input grid for counter example
#= z = Vector{Float64}(undef,3)

P = [[0;;0;;;0;;1;;;1;;0],[0;;;1;;;0],[0;;;0;;;1]]
R = [[0 0],[-.5;;],[.5;;]]
X = [Matrix{Float64}(undef,2,2),Matrix{Float64}(undef,1,1),Matrix{Float64}(undef,1,1)]
Y = [Matrix{Float64}(undef,2,2),Matrix{Float64}(undef,1,1),Matrix{Float64}(undef,1,1)]

genv = Gurobi.Env()

out = [Ψ!(z,X,Y,[x,-y,y],P,R,.6,genv) for y ∈ -.5:.01:1.75, x ∈ -1:.01:1]
serialize("Paper/new test data/counter_plot.ary", out) =#

out = deserialize("Paper/new test data/counter_plot.ary")
c = contour(-1:.01:1,-.5:.01:1.75, out, levels = 0:.25:4, legend = :bottomleft, xlabel = L"v_1", ylabel = L"-v_2 , v_3")

plot!([√.5,.75],[0,1.25], linestyle = :dash, label = false, linecolor = :green)
scatter!([√.5],[0], markershape = :circle, markercolor = :blue, markersize = 5, label = L"v^0")
scatter!([.75],[1.25], markershape = :square, markercolor = :green, markersize = 5, label = L"v^0 + d^0")
scatter!([-.75],[1.25], markershape = :star, markercolor = :red, markersize = 5, label = L"v^\star")

