using JuMP, HiGHS

@testset "Simple L1" begin
  # Test case 1
  z = [0.5, 0.2, 0.9, 0.1]
  p̄ = [0.25, 0.25, 0.25, 0.25]
  ξ = 0.5

  # Run the function
  popt, obj = worstcase_l1(z, p̄, ξ)

  # Display the results
  #println("Test case 1")
  #println("Optimal p: ", popt)
  #println("Objective value: ", obj)

  @test obj ≤ z' * p̄
  @test obj ≥ minimum(z)
end

@testset "L1 values" begin
  q = [0.4, 0.3, 0.1, 0.2]
  z = [1.0, 2.0, 5.0, 4.0]

  t = 0.0
  w = worstcase_l1(z, q, t)[2]
  @test w ≈ 2.3

  t = 1.0
  w = worstcase_l1(z, q, t)[2]
  @test w ≈ 1.1

  t = 2.0
  w = worstcase_l1(z, q, t)[2]
  @test w ≈ 1

  q1 = [1.0]
  z1 = [2.0]

  t = 0.0
  w = worstcase_l1(z1, q1, t)[2]
  @test w ≈ 2.0

  t = 0.01
  w = worstcase_l1(z1, q1, t)[2]
  @test w ≈ 2.0

  t = 1.0
  w = worstcase_l1(z1, q1, t)[2]
  @test w ≈ 2.0

  t = 2.0
  w = worstcase_l1(z1, q1, t)[2]
  @test w ≈ 2.0
end

@testset "L1 Against HiGHS" begin
  function JUMP_worstcase_l1(x, pbar, ξ)
    m = Model(HiGHS.Optimizer)
    set_optimizer_attribute(m, "log_to_console", false)
    @variable(m, p[1:length(x)])
    @variable(m, t[1:length(x)])
    @constraint(m, p .- pbar <= t)
    @constraint(m, pbar .- p <= t)
    @constraint(m, sum(t) <= ξ)
    @constraint(m, sum(p) == 1)
    @constraint(m, p .>= 0)
    @objective(m, Min, p' * x)
    optimize!(m)
    termination_status(m) != MOI.OPTIMAL && Error("JuMP failed to find optimal solution")
    return value.(p), value.(p)' * x
  end
  function JUMP_worstcase_l1_weighted(x, pbar, ξ, w)
    m = Model(HiGHS.Optimizer)
    set_optimizer_attribute(m, "log_to_console", false)
    @variable(m, p[1:length(x)])
    @variable(m, t[1:length(x)])
    @constraint(m, p .- pbar <= t)
    @constraint(m, pbar .- p <= t)
    @constraint(m, w' * t <= ξ)
    @constraint(m, sum(p) == 1)
    @constraint(m, p .>= 0)
    @objective(m, Min, p' * x)
    optimize!(m)
    termination_status(m) != MOI.OPTIMAL && Error("JuMP failed to find optimal solution")
    return value.(p), value.(p)' * x
  end
  for n in range(1, 100, step=10)
    x = randn(n)
    probs = rand(n)
    probs = probs / sum(probs)
    α = clamp(randn(Float64), 0, 2)
    @test JUMP_worstcase_l1(x, probs, α)[2] ≈ worstcase_l1(x, probs, α)[2]
    @test JUMP_worstcase_l1_weighted(x, probs, α, ones(n))[2] ≈ JUMP_worstcase_l1(x, probs, α)[2]
    # TODO: test weighted after it is implemented
    # @test JUMP_worstcase_l1_weighted(x, probs, α, ones(n))[2] ≈ worstcase_l1_w(x, probs, ones(n), α)[2]
    # w = abs.(rand(n))
    # w = w / sum(w)
    # @test JUMP_worstcase_l1_weighted(x, probs, α, w)[2] ≈ worstcase_l1_w(x, probs, w, α)[2]
  end
end
