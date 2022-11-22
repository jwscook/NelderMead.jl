module NelderMead

include("Vertexes.jl")
include("Simplexes.jl")

const Container = Union{AbstractVector, Tuple}

"""
    optimise(f, initial_vertex_positions; kwargs...)

Find minimum of function, `f`, first creating a Simplex from vertices at
`initial_vertex_positions`, and options passed in via kwargs.
"""
function optimise(f::T, initial_vertex_positions::U; kwargs...
   ) where {T<:Function, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}
  return optimise!(Simplex(f, initial_vertex_positions), f; kwargs...)
end

"""
    optimise(f, initial_positions, initial_step; kwargs...)

Find minimum of function, `f`, first creating a Simplex using a starting
vertex position, `initial_position`, and other vertices `initial_step` away from
that point in all directions, and options passed in via kwargs.
"""
function optimise(f::T, initial_position::Container, initial_step::Container;
    kwargs...) where {T<:Function}
  return optimise!(Simplex(f, initial_position, initial_step), f; kwargs...)
end

"""
    optimise!(s, f; kwargs...)

Find minimum of function, `f`, starting from Simplex, `s`, with options
passed in via kwargs.

# Keyword Arguments
-  stopval (default -Inf): stopping criterion when function evaluates
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

# Returns
`optimise` returns a tuple consisting of:
- minimiser: the location of the minimum
- minimumvalue: the value at the minimum
- returncode: the return code symbol
- numiters: the number of iterations taken
- simplex: the simplex in its final state (useful for restarting)
"""
function optimise!(s::Simplex{T,U}, f::F; kwargs...) where {F<:Function, T<:Real, U}
  kwargs = Dict(kwargs)
  stopval = get(kwargs, :stopval, -T(Inf))
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

  reflect(this, other) = Vertex(newposition(this, α, other), f)
  expand(this, other) = Vertex(newposition(this, -γ, other), f)
  contract(this, other) = Vertex(newposition(this, -β, other), f)
  shrink(this, other) = Vertex(newposition(this, δ, other), f)

  function shrink!(s::Simplex)
    lengthbefore = length(s)
    best = bestvertex(s)
    newvertices = [shrink(best, v) for v ∈ s if !isequal(v, best)]
    remove!(s, findall(v->!isequal(v, best), s.vertices))
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
      reflected = reflect(centroidposition(s), worst)

      if any(h->isequal(h, reflected), history)
        returncode = :ENDLESS_LOOP
        break
      end
      history .= circshift(history, 1)
      history[1] = reflected

      if best <= reflected < secondworst
        swapworst!(s, reflected)
      else
        centroid = Vertex(centroidposition(s), f)
        if reflected < best
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
      end
      returncode = assessconvergence(s,
        xtol_abs, xtol_rel, ftol_abs, ftol_rel, stopval)
    end
  end

  iters == maxiters && (returncode = :MAXITERS_REACHED)
  best = bestvertex(s)
  return (position(best), value(best), returncode, iters, s)
end # optimise

end
