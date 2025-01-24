using JuMP, Gurobi, MDPs, LinearAlgebra




"""
Computes the worst case distribution with a bounded total deviation `ξ`
from the underlying probability distribution `p̄` for the random variable `z`.

Efficiently computes the solution of:
min_p   p^T * z
s.t.    || p - p̄ ||_1  ≤ ξ
        1^T p = 1
        p ≥ 0

Notes
-----
This implementation works in O(n log n) time because of the sort. Using
quickselect to choose the right quantile would work in O(n) time.

This function does not check whether the provided probability distribution sums
to 1.

Returns
-------
Optimal solution `p` and the objective value
"""
function worstcase_l1(z::Vector{<:Real}, p̄::Vector{<:Real}, ξ::Real)
    (maximum(p̄) ≤ 1 + 1e-9 && minimum(p̄) ≥ -1e-9)  ||
        "values must be between 0 and 1"
    ξ ≥ zero(ξ)|| "ξ must be nonnegative"
    (length(z) > 0 && length(z) == length(p̄)) ||
            "z's values needs to be same length as p̄'s values"
    
    ξ = clamp(ξ, 0, 2)
    size = length(z)
    sorted_ind = sortperm(z)

    out = copy(p̄)       #duplicate it
    k = sorted_ind[1]   #index begins at 1

    ϵ = min(ξ / 2, 1 - p̄[k])
    out[k] += ϵ
    i = size

    while ϵ > 0 && i > 0
        k = sorted_ind[i]
        i -= 1
        difference = min(ϵ, out[k])
        out[k] -= difference
        ϵ -= difference
    end

    return out, out'*z
end



function robust_bellman_solve(P̄ₛ,W,Z,κ, env)
    lpm = Model(()->(Gurobi.Optimizer(env)))
    set_silent(lpm)
    set_attribute(lpm, "Threads", 1)
    set_attribute(lpm, "BarConvTol", 1e-9)
    set_attribute(lpm, "OptimalityTol", 1e-9)
    n = size(P̄ₛ)[2]
    m = size(P̄ₛ)[1]
    @variable(lpm, d[1:m])
    @variable(lpm, x[1:m])
    @variable(lpm, λ)
    @variable(lpm, yᵖ[1:n,1:m])
    @variable(lpm, yⁿ[1:n,1:m])
    @objective(lpm,Max,sum([x[a] + P̄ₛ[a,:]'*(yⁿ[:,a] .- yᵖ[:,a]) for a ∈ 1:m]) - κ*λ)
    @constraint(lpm, sum(d) == 1)
    @constraint(lpm, d .≥ 0)
    s = @constraint(lpm, -yᵖ .+ yⁿ .+ ones(n)*x' .≤ Z'*diagm(d))
    @constraint(lpm, yᵖ .+ yⁿ .- λ*W' .≤ 0)
    @constraint(lpm,yᵖ .≥ 0)
    @constraint(lpm,yⁿ .≥ 0)
    @constraint(lpm,λ ≥ 0)
    optimize!(lpm)
    return (πₛ = value.(d), pₛ = -dual.(s)')
end

function worst_prob_solve(P̄ₛ,W,Z,κ,d,env)
    lpm = Model(()->(Gurobi.Optimizer(env)))
    set_silent(lpm)
    set_optimizer_attribute(lpm, "Threads", 1)
    n = size(P̄ₛ)[2]
    m = size(P̄ₛ)[1]
    @variable(lpm, P[1:m,1:n])
    @variable(lpm, Θ[1:m,1:n])
    @objective(lpm,Min,sum([d[a]*P[a,:]'*Z[a,:] for a ∈ 1:m]))
    @constraint(lpm, P*ones(n) .== 1)
    @constraint(lpm, P .≥ 0)
    @constraint(lpm, Θ .≥ 0)
    @constraint(lpm, P .- P̄ₛ .≥ -Θ)
    @constraint(lpm, P̄ₛ .- P .≥ -Θ)
    @constraint(lpm, -sum([W[a,:]'*Θ[a,:] for a ∈ 1:m]) ≥ -κ)
    optimize!(lpm)
    return (πₛ = d, pₛ = value.(P))
end


function Bv!(vᵏ⁺¹,vᵏ,P̄,R,W,ξ,γ,env)
    for s ∈ eachindex(R)
        Z = R[s] + γ*ones(size(R[s])[1])*vᵏ'
        out = robust_bellman_solve(P̄[s],W[s],Z,ξ,env)
        vᵏ⁺¹[s] = [out.pₛ[a,:]'*Z[a,:] for a ∈ 1:size(R[s])[1]]'*out.πₛ
    end
end

function B!(vᵏ⁺¹,πᵏ,Pᵏ,vᵏ,P̄,R,W,γ,ξ,env)
    for s ∈ eachindex(R)
        Z = R[s] .+ γ*ones(size(R[s])[1])*vᵏ'
        out = robust_bellman_solve(P̄[s],W[s],Z,ξ,env)
        vᵏ⁺¹[s] = [out.pₛ[a,:]'*Z[a,:] for a ∈ 1:size(R[s])[1]]'*out.πₛ
        πᵏ[s] = out.πₛ
        Pᵏ[s] = out.pₛ
    end 
end

function Bμ!(vᵏ⁺¹,πᵏ,Pᵏ,vᵏ,P̄,R,W,γ,ξ,env)
    for s ∈ eachindex(R)
        Z = R[s] + γ*ones(size(R[s])[1])*vᵏ'
        out = worst_prob_solve(P̄[s],W[s],Z,ξ,πᵏ[s],env)
        vᵏ⁺¹[s] = [out.pₛ[a,:]'*Z[a,:] for a ∈ 1:size(R[s])[1]]'*out.πₛ
        Pᵏ[s] = out.pₛ
    end
end

function P_π!(P_π,πᵏ,Pᵏ)
    for s ∈ eachindex(Pᵏ)
        P_π[s,:] = Pᵏ[s]'*πᵏ[s]
    end
end

function R_π!(R_π,πᵏ,Pᵏ,R)
    for s ∈ eachindex(R)
        r = R[s]
        R_π[s] = [r[a,:]'*Pᵏ[s][a,:] for a ∈ 1:size(r)[1]]'*πᵏ[s]
    end
end



function VI(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = copy(v₀)
    Bv!(u,v,P̄,R,W,ξ,γ,env)
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        Bv!(u,v,P̄,R,W,ξ,γ,env)
    end
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end


function PAI(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = copy(v₀)
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        u .= (I - γ*P_π) \ R_π
    end
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end

function Ψ!(z,v,P̄,R,W,ξ,γ,env)
    Bv!(z,v,P̄,R,W,ξ,γ,env)
    return norm(z - v)
end

function Ψ∞!(z,v,P̄,R,W,ξ,γ,env)
    Bv!(z,v,P̄,R,W,ξ,γ,env)
    return norm(z - v, Inf)
end


function Filar(model,γ,ξ,W,ϵ,env,time_limit,η,β,v₀=zeros(state_count(model)))
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = copy(v₀)
    z = Vector{Float64}(undef,length(v₀))
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    s = zeros(state_count(model))
    while Ψ!(z,u,P̄,R,W,ξ,γ,env) > ϵ && time()-start < time_limit
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        s .= (I - γ*P_π) \ R_π - v
        α = 1
        δ = ((γ*P_π - I)'*(u-v))'*s
        while Ψ!(z,v+α*s,P̄,R,W,ξ,γ,env) - Ψ!(z,v,P̄,R,W,ξ,γ,env) > η*α*δ && time()-start < time_limit
            α *= β
        end
        u .= v + α*s
    end
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end

function Keiths(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = copy(v₀)
    w = copy(v₀)
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        d = norm(u-v,Inf)
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        w .= (I - γ*P_π) \ R_π
        while Ψ∞!(u,w,P̄,R,W,ξ,γ,env) > d && time()-start < time_limit
            w .= u
        end
    end
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end


function KeithMarek(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = copy(v₀)
    w = copy(v₀)
    z = Vector{Float64}(undef,length(v₀))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        d = norm(u-v,Inf)
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        w .= (I - γ*P_π) \ R_π
        if Ψ∞!(z,w,P̄,R,W,ξ,γ,env) > γ*d
            v .= u
            B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        else
            u .= w
        end
    end
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end

"""

Optimizes over values where `B u .≥ u`
"""
function Mareks(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,β,v₀=zeros(state_count(model)))
    # TODO: check if v₀ satisfies the LE condition?
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end
    
    v = copy(v₀)
    u = copy(v₀)
    for s ∈ 1:state_count(model)
        v[s] = -abs((1/(1-γ))*minimum(R[s]))
    end
    u .= v

    # Bellman operator
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)

    # For policy evaluation
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    s = zeros(state_count(model))

    # A temporary variable to check monotonicity
    Bu = zeros(state_count(model))
    
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)

        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)

        s .= (I - γ*P_π) \ R_π - v
        α = 1
        # TODO: should this be a u or v?
        Bu .= u
        B!(Bu,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        while any(u .> Bu) && time()-start < time_limit
            α *= β
            u .= v + α*s
            Bu .= u
            B!(Bu,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        end
    end
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end

function Winnicki(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,H,m,v₀=zeros(state_count(model)))
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = copy(v₀)
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        for i = 1:H
            B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        end
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        for i = 1:m
            u .= R_π + γ*P_π*u
        end
    end
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end

function HoffKarp(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = copy(v₀)
    w = zeros(state_count(model))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        Bμ!(w,πᵏ,Pᵏ,u,P̄,R,W,γ,ξ,env)
        while norm(w-u,Inf) > 1e-6 && time()-start < time_limit
            P_π!(P_π,πᵏ,Pᵏ)
            R_π!(R_π,πᵏ,Pᵏ,R)
            u .= (I - γ*P_π) \ R_π
            Bμ!(w,πᵏ,Pᵏ,u,P̄,R,W,γ,ξ,env)
        end 
    end
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end

function PPI(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,β,ϵ₂,v₀=zeros(state_count(model)))
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = copy(v₀)
    w = zeros(state_count(model))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        Bμ!(w,πᵏ,Pᵏ,u,P̄,R,W,γ,ξ,env)
        while norm(w-u,Inf) > ϵ₂ && time()-start < time_limit
            P_π!(P_π,πᵏ,Pᵏ)
            R_π!(R_π,πᵏ,Pᵏ,R)
            u .= (I - γ*P_π) \ R_π
            Bμ!(w,πᵏ,Pᵏ,u,P̄,R,W,γ,ξ,env)
        end 
        ϵ₂ *= β
    end
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end