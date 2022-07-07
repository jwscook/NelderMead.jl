using Random, NelderMead, Test
import NelderMead: Vertex, Simplex

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

    @testset "Errors are caught" begin
      N = 2
      @test_throws ArgumentError NelderMead.optimise(x->objective(x, N), [rand(1),
                                                  rand(2), rand(3)])
      @test_throws ArgumentError NelderMead.optimise(x->objective(x, N), rand(2),
                                                  rand(3))
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

    @testset "hash, isequal, ==" begin
      v0 = Vertex([0.0], 0)
      v1 = Vertex([1.0], 1)
      @test !(v0 == v1)
      @test !isequal(v0, v1)
      @test hash(v0) != hash(v1)
      s01 = Simplex([v0, v1])
      @test s01 == s01
      @test isequal(s01, s01)
      @test hash(s01) == hash(s01)
      v2 = deepcopy(v1)
      v2.position[1] += rand()
      @test v1 != v2
      @test !(v1 == v2)
      @test hash(v1) != hash(v2)
      s02 = deepcopy(s01)
      v1.position[1] += rand()
      @test !(s02 == s01)
      @test hash(s02) != hash(s01)
      v3 = NelderMead.Vertex([1.0], 2)
      @test Simplex([v0, v1]) != Simplex([v0, v3])
      @test !isequal(Simplex([v0, v1]), Simplex([v0, v3]))
      @test hash(Simplex([v0, v1])) != hash(Simplex([v0, v3]))
    end

  end
end
