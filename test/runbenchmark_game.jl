include("../src/MarkovGames.jl")
using DataFrames, CSV, Gurobi


function time_algorithm(alg,P,R,γ,ϵ,env,time_limit)
    if alg.name == "VI"
        return @elapsed VI(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "PAI"
        return @elapsed PAI(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "HK"
        return @elapsed HoffKarp(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "FT"
        return @elapsed Filar(P,R,γ,ϵ,env,time_limit,alg.η,alg.β)
    elseif alg.name == "M1"
        return @elapsed Mareks(P,R,γ,ϵ,env,time_limit,alg.β)
    elseif alg.name == "K1"
        return @elapsed Keiths(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "KM"
        return @elapsed KeithMarek(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "WIN"
        return @elapsed Winnicki(P,R,γ,ϵ,env,time_limit,alg.H,alg.m)
    elseif alg.name == "PPI"
        return @elapsed PPI(P,R,γ,ϵ,env,time_limit,alg.ϵ₂,alg.β)
    else error("algorithm name must be one of: \n
                VI, PAI, HK, PPI, FT, M1, K1, KM, WIN")
    end
end

function benchmark_run(state_nums::Vector{Int64}, Γ::Vector{Float64}, algs, action_nums::Vector{Int64}, r_lower::Float64, r_upper::Float64, η::Number, ϵ::Number, maxtime::Float64)
    results = DataFrame(time = Vector{Float64}(undef,0), state_number = Vector{Int64}(undef,0), algorithm = Vector{String}(undef,0), γ = Vector{Float64}(undef,0))
    G_ENV = Gurobi.Env()
    frame_lock = ReentrantLock()
    Threads.@threads for nₛ ∈ state_nums
        for disc ∈ Γ
            G = make_markov_game(nₛ,rand(action_nums,nₛ),rand(action_nums,nₛ),r_lower,r_upper,η)
            for alg ∈ algs
                t = time_algorithm(alg,G.transition,G.rewards,disc,ϵ,G_ENV,maxtime)
                @lock frame_lock push!(results, [t,nₛ, alg.name,disc]) 
            end
        end
    end
    return results
end


#= algorithms = [(name = "VI",), (name = "PAI",), (name = "HK",), (name = "FT", η = 1e-3, β = .5),
              (name = "K1",), (name = "KM",), (name = "WIN", H = 10, m = 100),
              (name = "PPI", ϵ₂ = .1, β = .5)]  =#
#= algorithms = [(name = "PAI",), (name = "K1",), (name = "KM",)]  =#
