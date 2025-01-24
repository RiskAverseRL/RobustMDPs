using JuMP, LinearAlgebra, Distributions, StatsBase, Random, BenchmarkTools, Gurobi

function make_markov_game(num_states::Int64,num_actions_x::Vector{Int64},num_actions_y::Vector{Int64},r_lower::Float64,r_upper::Float64, η::Float64)
    num_next = Int(round(η*num_states))
    P = [zeros(num_actions_y[s],num_actions_x[s],num_states) for s ∈ 1:num_states]
    R = [rand(Uniform(r_lower,r_upper),num_actions_y[s],num_actions_x[s]) for s ∈ 1:num_states]
    for s ∈ 1:num_states
        for a ∈ 1:num_actions_y[s]
            for b ∈ 1:num_actions_x[s]  
                P[s][a,b,shuffle(1:num_states)[1:num_next]] = normalize(rand(Exponential(1),num_next),1)
            end
        end
    end
    return (transition = P, rewards = R)
end


function matrix_game_solve(A, env)
    lpm = Model(()->(Gurobi.Optimizer(env)))
    set_silent(lpm)
    set_optimizer_attribute(lpm, "Threads", 1)
    n = size(A)[2]
    @variable(lpm, u)
    @variable(lpm, x[1:n])
    @objective(lpm,Min,u)
    s = @constraint(lpm, A*x .- u ≤ 0)
    @constraint(lpm,sum(x) == 1)
    @constraint(lpm, x ≥ 0)
    optimize!(lpm)
    return (x = value.(x),y = -dual(s),u = value(u))
end


function Bv!(vᵏ⁺¹,vᵏ,P,R,γ,env)
    for s ∈ eachindex(R)
        vᵏ⁺¹[s] = matrix_game_solve(R[s]+γ*sum([P[s][:,:,s2]*vᵏ[s2] for s2 ∈ eachindex(P)]),env).u
    end
end

function B!(vᵏ⁺¹,X,Y,vᵏ,P,R,γ,env) 
    for s ∈ eachindex(R)
        out = matrix_game_solve(R[s]+γ*sum([P[s][:,:,s2]*vᵏ[s2] for s2 ∈ eachindex(P)]),env)
        vᵏ⁺¹[s] = out.u
        X[s] = out.x
        Y[s] = out.y
    end 
end

function Bμ!(vᵏ⁺¹,X,Y,vᵏ,P,R,γ)
    for s ∈ eachindex(R)
        Y[s] = zeros(size(R[s])[1])
        max = findmax((R[s]+γ*sum([P[s][:,:,s2]*vᵏ[s2] for s2 ∈ eachindex(P)]))*X[s])
        Y[s][max[2]] = 1
        vᵏ⁺¹[s] = max[1]
    end
end

function P_π!(P_π,X,Y,P)
    for s ∈ eachindex(P)
        P_π[s,:] = [Y[s]'*P[s][:,:,s_next]*X[s] for s_next ∈ eachindex(P)]
    end
end

function R_π!(R_π,X,Y,R)
    for s ∈ eachindex(R)
        R_π[s] = Y[s]'*R[s]*X[s]
    end
end

function VI(P,R,γ,ϵ,env,v₀=zeros(length(R)))
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    v = copy(v₀)
    u = copy(v₀)
    Bv!(u,v,P,R,γ,env)
    while norm(u-v, Inf) > ϵ
        v .= u
        Bv!(u,v,P,R,γ,env)
    end
    B!(u,X,Y,v,P,R,γ,env)
    return (value = v, x = X, y = Y)
end


function PAI(P,R,γ,ϵ,env,v₀=zeros(length(R)))
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    v = copy(v₀)
    u = copy(v₀)
    B!(u,X,Y,v,P,R,γ,env)
    P_π = zeros(length(P),length(P))
    R_π = zeros(length(R))
    while norm(u-v, Inf) > ϵ
        v .= u
        B!(u,X,Y,v,P,R,γ,env)
        P_π!(P_π,X,Y,P)
        R_π!(R_π,X,Y,R)
        u .= (I - γ*P_π) \ R_π
    end
    return (value = v, x = X, y = Y)
end

function Ψ!(z,v,P,R,γ,env)
    Bv!(z,v,P,R,γ,env)
    return norm(z - v)
end

function Ψ∞!(z,v,P,R,γ,env)
    Bv!(z,v,P,R,γ,env)
    return norm(z - v, Inf)
end


function Filar(P,R,γ,ϵ,env,η,β,v₀=zeros(length(R)))
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    v = copy(v₀)
    u = copy(v₀)
    z = Vector{Float64}(undef,length(v₀))
    P_π = zeros(length(P),length(P))
    R_π = zeros(length(R))
    s = zeros(length(R))
    while Ψ!(z,u,P,R,γ,env) > ϵ
        v .= u
        B!(u,X,Y,v,P,R,γ,env)
        P_π!(P_π,X,Y,P)
        R_π!(R_π,X,Y,R)
        s .= (I - γ*P_π) \ R_π - v
        α = 1
        δ = ((γ*P_π - I)'*(u-v))'*s
        while Ψ!(z,v+α*s,P,R,γ,env) - Ψ!(z,v,P,R,γ,env) > η*α*δ
            α *= β
        end
        u .= v + α*s
    end
    return (value = u, x = X, y = Y)
end

function Keiths(P,R,γ,ϵ,env,v₀=zeros(length(R)))
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    v = copy(v₀)
    u = copy(v₀)
    w = copy(v₀)
    B!(u,X,Y,v,P,R,γ,env)
    P_π = zeros(length(P),length(P))
    R_π = zeros(length(R))
    while norm(u-v, Inf) > ϵ
        v .= u
        B!(u,X,Y,v,P,R,γ,env)
        d = norm(u-v,Inf)
        P_π!(P_π,X,Y,P)
        R_π!(R_π,X,Y,R)
        w .= (I - γ*P_π) \ R_π
        while Ψ∞!(u,w,P,R,γ,env) > d
            w .= u
        end
    end
    B!(u,X,Y,v,P,R,γ,env)
    return (value = u, x = X, y = Y)
end


function KeithMarek(P,R,γ,ϵ,env,v₀=zeros(length(R)))
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    v = copy(v₀)
    u = copy(v₀)
    w = copy(v₀)
    z = Vector{Float64}(undef,length(v₀))
    B!(u,X,Y,v,P,R,γ,env)
    P_π = zeros(length(P),length(P))
    R_π = zeros(length(R))
    while norm(u-v, Inf) > ϵ
        v .= u
        B!(u,X,Y,v,P,R,γ,env)
        d = norm(u-v,Inf)
        P_π!(P_π,X,Y,P)
        R_π!(R_π,X,Y,R)
        w .= (I - γ*P_π) \ R_π
        if Ψ∞!(z,w,P,R,γ,env) > γ*d
            v .= u
            B!(u,X,Y,v,P,R,γ,env)
        else
            u .= w
        end
    end
    return (value = v, x = X, y = Y)
end

"""

Optimizes over values where `B u .≥ u`
"""
function Mareks(P,R,γ,ϵ,env,β,v₀=zeros(length(R)))
    # TODO: check if v₀ satisfies the LE condition?
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    
    v = copy(v₀)
    u = copy(v₀)
    for s ∈ eachindex(R)
        v[s] = -abs((1/(1-γ))*minimum(R[s]))
    end
    u .= v

    # Bellman operator
    B!(u,X,Y,v,P,R,γ,env)

    # For policy evaluation
    P_π = zeros(length(P),length(P))
    R_π = zeros(length(R))
    s = zeros(length(R))

    # A temporary variable to check monotonicity
    Bu = zeros(length(R))
    
    while norm(u-v, Inf) > ϵ
        v .= u
        B!(u,X,Y,v,P,R,γ,env)

        P_π!(P_π,X,Y,P)
        R_π!(R_π,X,Y,R)

        s .= (I - γ*P_π) \ R_π - v
        α = 1
        # TODO: should this be a u or v?
        Bu .= u
        B!(Bu,X,Y,v,P,R,γ,env)
        while any(u .> Bu)
            α *= β
            u .= v + α*s
            Bu .= u
            B!(Bu,X,Y,v,P,R,γ,env)
        end
    end
    return (value = u, x = X, y = Y)
end

function Winnicki(P,R,γ,ϵ,env,H,m,v₀=zeros(length(R)))
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    v = copy(v₀)
    u = copy(v₀)
    B!(u,X,Y,v,P,R,γ,env)
    P_π = zeros(length(P),length(P))
    R_π = zeros(length(R))
    while norm(u-v, Inf) > ϵ
        v .= u
        for i = 1:H
            B!(u,X,Y,v,P,R,γ,env)
        end
        P_π!(P_π,X,Y,P)
        R_π!(R_π,X,Y,R)
        for i = 1:m
            u .= R_π + γ*P_π*u
        end
    end
    return (value = v, x = X, y = Y)
end

function HoffKarp(P,R,γ,ϵ,env,v₀=zeros(length(R)))
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    v = copy(v₀)
    u = copy(v₀)
    w = zeros(length(R))
    B!(u,X,Y,v,P,R,γ,env)
    P_π = zeros(length(P),length(P))
    R_π = zeros(length(R))
    while norm(u-v, Inf) > ϵ
        v .= u
        B!(u,X,Y,v,P,R,γ,env)
        Bμ!(w,X,Y,u,P,R,γ)
        while norm(w-u,Inf) > 0
            u .= w
            P_π!(P_π,X,Y,P)
            R_π!(R_π,X,Y,R)
            w .= (I - γ*P_π) \ R_π
            Bμ!(w,X,Y,u,P,R,γ)
        end 
    end
    B!(u,X,Y,v,P,R,γ,env)
    return (value = u, x = X, y = Y)
end

function PPI(P,R,γ,ϵ,env,ϵ₂,β,v₀=zeros(length(R)))
    X = [zeros(size(R[s])[2]) for s ∈ eachindex(R)]
    Y = [zeros(size(R[s])[1]) for s ∈ eachindex(R)]
    v = copy(v₀)
    u = copy(v₀)
    w = zeros(length(R))
    B!(u,X,Y,v,P,R,γ,env)
    P_π = zeros(length(P),length(P))
    R_π = zeros(length(R))
    while norm(u-v, Inf) > ϵ
        v .= u
        B!(u,X,Y,v,P,R,γ,env)
        Bμ!(w,X,Y,u,P,R,γ)
        while norm(w-u,Inf) > ϵ₂
            u .= w
            P_π!(P_π,X,Y,P)
            R_π!(R_π,X,Y,R)
            w .= (I - γ*P_π) \ R_π
            Bμ!(w,X,Y,u,P,R,γ)
        end
        ϵ₂ *= β
    end
    B!(u,X,Y,v,P,R,γ,env)
    return (value = u, x = X, y = Y)
end




#=
Big Match:
P = [[1;0 ;; 1;0 ;;; 0;0 ;; 0;1 ;;; 0;1 ;; 0;0],[0;;;1;;;0], [0;;;0;;;1]]
R = [[0 1; 1 0], [0;;], [1;;]]
Vand der Wal Counter-Example:
P = [[1.0 0.3333333333333333; 1.0 1.0;;; 0.0 0.6666666666666666; 0.0 0.0],[0.0;;; 1.0]]
R = [[3.0 6.0; 2.0 1.0],[0.0;;]]
γ=.75
Filar Counter-Example:
P = [[0;;0;;;0;;1;;;1;;0],[0;;;1;;;0],[0;;;0;;;1]]
R = [[0 0],[-.5;;],[.5;;]]
=#

#benchmark_run(100*ones(20),[(name = "VI", ϵ = 1e-7),(name = "PAI", ϵ = 1e-7),(name = "HK", ϵ = 1e-7),(name = "FT", ϵ = 1e-7, η = .001, β = .5),(name = "M1", ϵ = 1e-7, β = .5),(name = "K1", ϵ = 1e-7),(name = "KM", ϵ = 1e-7),(name = "WIN", ϵ = 1e-7, H = 10, m = 100),(name = "PPI", ϵ = 1e-7, ϵ₂ = .1, β = .5)], [1,2,3,5,10], -3.0, 5.0, .2, .9)