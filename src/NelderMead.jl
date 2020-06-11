module NelderMead

include("Vertexes.jl")
include("Simplexes.jl")

"""
    solve(f, initial_vertex_positions; kwargs...)

Find minimum of function, `f`, first creating a Simplex from vertices at
`initial_vertex_positions`, and options passed in via kwargs.
"""
function solve(f::T, initial_vertex_positions::U; kwargs...
   ) where {T<:Function, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}
  return solve(f, Simplex(f, initial_vertex_positions); kwargs...)
end

"""
    solve(f, initial_positions, initial_step; kwargs...)

Find minimum of function, `f`, first creating a Simplex using a starting
vertex position, `initial_position`, and other vertices `initial_step` away from
that point in all directions, and options passed in via kwargs.
"""
function solve(f::T, initial_position::AbstractVector{U},
               initial_step::AbstractVector{V}; kwargs...
               ) where {T<:Function, U<:Number, V<:Number}
  return solve(f, Simplex(f, initial_position, initial_step); kwargs...)
end

"""
    solve(f, lower, upper, gridsize; kwargs...)

Find minimum of function, `f`, first creating a `2 .* prod(gridsize)` Simplices
within a hyper-rectangle spanning from `lower` to `upper`, and
options passed in via kwargs.
"""
function solve(f::T, lower::AbstractVector{U}, upper::AbstractVector{U},
    gridsize::AbstractVector{<:Integer}; kwargs...) where {T<:Function, U<:Number}

  all(gridsize .> 0) || throw(ArgumentError("gridsize, $gridsize, must .> 0"))
  if !(length(lower) == length(upper) == length(gridsize))
    throw(ArgumentError("The lengths of lower $lower, upper $upper, and
                        gridsize $gridsize must be the same"))
  end
  if !all(lower .< upper)
    throw(ArgumentError("lower $lower must .< upper $upper"))
  end

  dim = length(lower)
  index2position(i) = (i .- 1) ./ gridsize .* (upper .- lower) .+ lower
  index2values = Dict()
  totaltime = @elapsed for ii ∈ CartesianIndices(Tuple(gridsize .+ 1))
    index = collect(Tuple(ii))
    x = index2position(index)
    index2values[index] = f(x)
  end
  simplices = Set{Simplex}()
  function generatesimplices!(simplices, direction)
    for ii ∈ CartesianIndices(Tuple(((gridsize .+ 1) .* ones(Int, dim))))
      vertices = Vector{Vector{U}}()
      index = collect(Tuple(ii))
      for i ∈ 1:dim + 1
        vertexindex = [index[j] + ((j == i) ? direction : 0) for j ∈ 1:dim]
        all(1 .<= vertexindex .<= gridsize .+ 1) || continue
        vertex = index2position(vertexindex)
        push!(vertices, vertex)
      end
      length(vertices) != dim + 1 && continue
      push!(simplices, Simplex(f, vertices))
    end
  end
  totaltime += @elapsed generatesimplices!(simplices, 1)
  totaltime += @elapsed generatesimplices!(simplices, -1)
  @assert length(simplices) == 2 * prod(gridsize)
  totaltime += @elapsed solutions = map(s->solve(f, s; kwargs...), collect(simplices))
  return solutions
end

"""
    solve(f, s; kwargs...)

Find minimum of function, `f`, starting from Simplex, `s`, with options
passed in via kwargs.

# Keyword Arguments
-  stopval (default sqrt(eps())): stopping criterion when function evaluates
equal to or less than stopval
-  xtol_abs (default zeros(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this absolute tolerance
-  xtol_rel (default eps(T)) .* ones(Bool, dimensionality(s)): stop if
the vertices of simplex get within this relative tolerance
-  ftol_abs (default zero(real(U))): stop if function evaluations at the
vertices are close to one another by this absolute tolerance
-  ftol_rel (default 1000eps(real(U))): stop if function evaluations at the
vertices are close to one another by this relative tolerance
-  maxiters (default 1000): maximum number of iterations of the Nelder Mead
algorithm
-  timelimit (default Inf): stop if it takes longer than this in seconds
-  α (default 1): Reflection factor
-  β (default 0.5): Contraction factor
-  γ (default 2): Expansion factor
-  δ (default 0.5): Shrinkage factor
"""
function solve(f::F, s::Simplex{T,U}; kwargs...) where {F<:Function, T<:Real, U}
  kwargs = Dict(kwargs)
  stopval = get(kwargs, :stopval, sqrt(eps()))
  xtol_abs = get(kwargs, :xtol_abs, zeros(T)) .* ones(Bool, dimensionality(s))
  xtol_rel = get(kwargs, :xtol_rel, eps(T)) .* ones(Bool, dimensionality(s))
  ftol_abs = get(kwargs, :ftol_abs, zero(real(U)))
  ftol_rel = get(kwargs, :ftol_rel, 1000eps(real(U)))
  maxiters = get(kwargs, :maxiters, 1000)
  timelimit = get(kwargs, :timelimit, Inf)
  α = get(kwargs, :α, 1)
  β = get(kwargs, :β, 0.5)
  γ = get(kwargs, :γ, 2)
  δ = get(kwargs, :δ, 0.5)

  @assert α >= 0 "$α >= 0"
  @assert 0 <= β < 1 "0 <= $β < 1"
  @assert γ > 1 "$γ > 1"
  @assert γ > α "$γ > $α"

  function Vertexlocal(centroid::Vertex{T, U}, ϵ::Number, other::Vertex{T, U}
      ) where {T, U}
    x = centroid + ϵ * (centroid - other)
    return Vertex(x, f(x))
  end
  reflect(centroid, other) = Vertexlocal(centroid, α, other)
  expand(centroid, other) = Vertexlocal(centroid, -γ, other)
  contract(centroid, other) = Vertexlocal(centroid, -β, other)

  function shrink!(s::Simplex)
    lengthbefore = length(s)
    best = bestvertex(s)
    newvertices = [Vertexlocal(best, -δ, v) for v ∈ s if !≡(v, best)]
    removelist = [v for v ∈ s if !≡(v, best)]
    for d ∈ removelist, v ∈ s
      ≡(d, v) && remove!(s, v)
    end
    push!(s, newvertices...)
    sort!(s, by=v->value(v))
    @assert length(s) == lengthbefore
    return nothing
  end

  iters, totaltime = 0, 0.0
  returncode = assessconvergence(s,
    xtol_abs, xtol_rel, ftol_abs, ftol_rel, stopval)
  history = deepcopy(s.vertices)
  while returncode == :CONTINUE && iters < maxiters && totaltime < timelimit
    totaltime += @elapsed begin
      iters += 1
      best = bestvertex(s)
      worst = worstvertex(s)
      secondworst = secondworstvertex(s)
      centroid = findcentroid(f, s)
      reflected = reflect(centroid, worst)

      if any(h->isequal(h, reflected), history)
        returncode = :ENDLESS_LOOP
        break
      end
      history[2:end-1] .= history[2:end-1]
      history[1] = reflected

      if best <= reflected < secondworst
        swapworst!(s, reflected)
      elseif reflected < best
        expanded = expand(centroid, reflected)
        expanded < reflected && swapworst!(s, expanded)
        expanded >= reflected && swapworst!(s, reflected)
      elseif secondworst <= reflected < worst
        contracted = contract(centroid, reflected)
        contracted <= reflected ? swapworst!(s, contracted) : shrink!(s)
      elseif reflected >= worst
        contracted = contract(centroid, worst)
        contracted < worst ? swapworst!(s, contracted) : shrink!(s)
      end
      returncode = assessconvergence(s,
        xtol_abs, xtol_rel, ftol_abs, ftol_rel, stopval)
    end
  end

  iters == maxiters && (returncode = :MAXITERS_REACHED)
  best = bestvertex(s)
  return value(best), position(best), returncode, iters
end # solve

end
