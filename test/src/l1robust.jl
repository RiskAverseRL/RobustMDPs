

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
    q = [ 0.4, 0.3, 0.1, 0.2 ]
    z = [ 1.0, 2.0, 5.0, 4.0 ]

    t = 0.
    w = worstcase_l1(z, q, t)[2]
    @test w ≈ 2.3

    t = 1.
    w = worstcase_l1(z, q, t)[2]
    @test w ≈ 1.1

    t = 2.
    w = worstcase_l1(z, q, t)[2]
    @test w ≈ 1

    q1 = [ 1.0 ]
    z1 = [ 2.0 ]

    t = 0.
    w = worstcase_l1(z1, q1, t)[2]
    @test w ≈ 2.0

    t = 0.01
    w = worstcase_l1(z1, q1, t)[2]
    @test w ≈ 2.0

    t = 1.
    w = worstcase_l1(z1, q1, t)[2]
    @test w ≈ 2.0

    t = 2.
    w = worstcase_l1(z1, q1, t)[2]
    @test w ≈ 2.0
end
