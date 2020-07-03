using Random, NelderMead, Test

Random.seed!(0)

@testset "NelderMead tests" begin
  @testset "ND Rosenbrock" begin
    rosenbrock1d(x) =  (1.0 - x[1])^2
    function rosenbrocknd(x)
      ithterm(i) = (1 - x[i])^2 + 100 * (x[i+1] - x[i]^2)^2
      return mapreduce(ithterm, +, 1:length(x)-1)
    end

    function test_solution(solution, N)
      minimum, minimiser, returncode, numiters = solution

      @test minimum <= stopval
      @test isapprox(ones(N), minimiser, rtol=1.0e-6)
      @test returncode == :STOPVAL_REACHED
    end

    stopval = eps()

    objective(x, N) = N == 1 ? rosenbrock1d(x) : rosenbrocknd(x)

    @testset "Single optimise - initial position and simplex size" begin
      for N ∈ 1:10
        solution = NelderMead.optimise(x->objective(x, N), zeros(N), ones(N),
                                      stopval=stopval, maxiters=100_000)
        test_solution(solution, N)
      end
    end
    @testset "Single optimise - vertex locations" begin
      for N ∈ 1:10
        solution = NelderMead.optimise(x->objective(x, N), [rand(N) for i ∈ 1:N+1],
                                    stopval=stopval, maxiters=100_000)
        test_solution(solution, N)
      end
    end
    @testset "Grid optimise" begin
      for N ∈ 1:3
        gridsize = [rand(1:3) for i ∈ 1:N]
        solutions = NelderMead.optimise(x->objective(x, N), zeros(N), ones(N), gridsize,
                                    stopval=stopval, maxiters=100_000)
        @assert length(solutions) >= 1
        test_solution.(solutions, N)
      end
    end
    @testset "Errors are caught" begin
      N = 2
      @test_throws ArgumentError NelderMead.optimise(x->objective(x, N), [rand(1),
                                                  rand(2), rand(3)])
      @test_throws ArgumentError NelderMead.optimise(x->objective(x, N), rand(2),
                                                  rand(3))
      @test_throws ArgumentError NelderMead.optimise(x->objective(x, N), [0.0],
                                                  [0.0], [1])
      @test_throws ArgumentError NelderMead.optimise(x->objective(x, N), rand(1),
                                                  rand(2), [1, 2, 3])
      @test_throws ArgumentError NelderMead.optimise(x->objective(x, N), rand(2),
                                                  rand(2), [0, 0])
    end

    @testset "Stretched grid causes endless loop" begin
      N = 2
      function stretchedobjective(x)
        x[2] /= 10.0
        return objective(x, N)
      end
      solution = NelderMead.optimise(stretchedobjective, zeros(N), ones(N),
                                  maxiters=1000, xtol_rel=2eps())
      _, _, returncode, numiters = solution
      @test returncode == :ENDLESS_LOOP
    end

    @testset "Nan causing endless loop" begin
      N = 2
      function nanobjective(x)
        output = objective(x, N)
        x[2] > 1.1 && (output += NaN)
        return output
      end
      solution = NelderMead.optimise(nanobjective, zeros(N), ones(N) / 2,
                                  maxiters=1000, xtol_rel=2eps())
      _, _, returncode, numiters = solution
      @test returncode == :ENDLESS_LOOP
    end

  end
end
