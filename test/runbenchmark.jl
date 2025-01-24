include("../src/l1robust.jl")
using DataFrames, CSV, Gurobi, Distributions, Random, MDPs

function make_random_inventory(state_number,sale_price_cap,order_capacity_cap)
    state_number ≥ 2 || error("There must be atleast 2 states")

    inventory_size = rand(1:state_number-1)
    backlog_size = state_number - inventory_size - 1
    maximum_order = rand(1:min(order_capacity_cap,inventory_size))

    sale_price = rand()*sale_price_cap
    item_cost = rand()*sale_price
    item_storage_cost = rand()*sale_price
    item_backlog_cost = rand()*sale_price
    delivery_cost = rand()*(maximum_order*(sale_price-item_cost))

    expected_demand = rand()*(state_number-1)
    demand_values = 0:(state_number-1)
    demand_dist = Poisson(expected_demand)
    demand_probabilities = map(x->pdf(demand_dist,x), demand_values)
    demand_probabilities[end] += max(1-sum(demand_probabilities),0)

    demand = Domains.Inventory.Demand(demand_values,demand_probabilities)
    costs = Domains.Inventory.Costs(item_cost,delivery_cost,item_storage_cost,item_backlog_cost)
    limits = Domains.Inventory.Limits(inventory_size,backlog_size,maximum_order)

    problem_params = Domains.Inventory.Parameters(demand,costs,sale_price,limits)
    return Domains.Inventory.Model(problem_params)
end

function make_random_grid(side_length)
    reward_s = zeros(side_length*side_length)
    reward_s[rand(1:side_length*side_length)] = 1
    pars = Domains.GridWorld.Parameters(reward_s,side_length,rand())
    return Domains.GridWorld.Model(pars)
end

function make_random_ruin(max_capital)
    return Domains.Gambler.Ruin(rand(),max_capital)
end


function time_algorithm(alg,model,γ,ξ,W,ϵ,env,max_time)
    time = 0
    if alg.name == "VI"
        time = @elapsed VI(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "PAI"
        time = @elapsed PAI(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "HK"
        time = @elapsed HoffKarp(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "FT"
        time = @elapsed Filar(model,γ,ξ,W,ϵ,env,max_time,alg.η,alg.β)
    elseif alg.name == "M1"
        time = @elapsed Mareks(model,γ,ξ,W,ϵ,env,max_time,alg.β)
    elseif alg.name == "K1"
        time = @elapsed Keiths(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "KM"
        time = @elapsed KeithMarek(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "WIN"
        time = @elapsed Winnicki(model,γ,ξ,W,ϵ,env,max_time,alg.H,alg.m)
    elseif alg.name == "PPI"
        time = @elapsed PPI(model,γ,ξ,W,ϵ,env,max_time,alg.β,alg.ϵ₂)
    else error("algorithm name must be one of: \n
                VI, PAI, HK, PPI, FT, M1, K1, KM, WIN")
    end
    return time ≥ max_time ? max_time : time
end

function benchmark_run_inventory(algs ,state_nums::Vector{Int64}, Γ::Vector{Float64}, ξ, ϵ::Number, maxtime::Float64)
    results = DataFrame(time = Vector{Float64}(undef,0), state_number = Vector{Int64}(undef,0), algorithm = Vector{String}(undef,0), γ = Vector{Float64}(undef,0))
    G_ENV = Gurobi.Env()
    frame_lock = ReentrantLock()
    Threads.@threads for nₛ ∈ state_nums
        for disc ∈ Γ
            inv_prob = make_random_inventory(nₛ,100,50)
            W = [ones(action_count(inv_prob,s),state_count(inv_prob)) for s ∈ 1:state_count(inv_prob)]
            for alg ∈ algs
                t = time_algorithm(alg,inv_prob,disc,ξ,W,ϵ,G_ENV, maxtime)
                @lock frame_lock push!(results, [t,nₛ, alg.name,disc]) 
            end
        end
    end
    return results
end

function benchmark_run_gridworld(algs ,state_nums::Vector{Int64}, Γ::Vector{Float64}, ξ, ϵ::Number, maxtime::Float64)
    results = DataFrame(time = Vector{Float64}(undef,0), state_number = Vector{Int64}(undef,0), algorithm = Vector{String}(undef,0), γ = Vector{Float64}(undef,0))
    G_ENV = Gurobi.Env()
    frame_lock = ReentrantLock()
    Threads.@threads for nₛ ∈ state_nums
        for disc ∈ Γ
            grid_prob = make_random_grid(nₛ)
            W = [ones(action_count(grid_prob,s),state_count(grid_prob)) for s ∈ 1:state_count(grid_prob)]
            for alg ∈ algs
                t = time_algorithm(alg,grid_prob,disc,ξ,W,ϵ,G_ENV, maxtime)
                @lock frame_lock push!(results, [t,nₛ^2, alg.name,disc]) 
            end
        end
    end
    return results
end

function benchmark_run_ruin(algs ,state_nums::Vector{Int64}, Γ::Vector{Float64}, ξ, ϵ::Number, maxtime::Float64)
    results = DataFrame(time = Vector{Float64}(undef,0), state_number = Vector{Int64}(undef,0), algorithm = Vector{String}(undef,0), γ = Vector{Float64}(undef,0))
    G_ENV = Gurobi.Env()
    frame_lock = ReentrantLock()
    Threads.@threads for nₛ ∈ state_nums
        for disc ∈ Γ
            ruin_prob = make_random_ruin(nₛ)
            W = [ones(action_count(ruin_prob,s),state_count(ruin_prob)) for s ∈ 1:state_count(ruin_prob)]
            for alg ∈ algs
                t = time_algorithm(alg,ruin_prob,disc,ξ,W,ϵ,G_ENV, maxtime)
                @lock frame_lock push!(results, [t,nₛ+1, alg.name,disc]) 
            end
        end
    end
    return results
end

#= algorithms = [(name = "VI",), (name = "PAI",), (name = "HK",), (name = "FT", η = 1e-3, β = .5),
              (name = "M1", β = .5), (name = "K1",), (name = "KM",), (name = "WIN", H = 10, m = 100),
              (name = "PPI", ϵ₂ = .1, β = .5)]  =#
#= algorithms = [(name = "PAI",), (name = "K1",), (name = "KM",)]  =#

#[ones(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]