struct Simplex{T<:Number, U}
  vertices::Vector{Vertex{T,U}}
end
function Simplex(f::T, ic::AbstractVector{U}, initial_step::AbstractVector{V}
    ) where {T<:Function, U<:Number, V<:Number}
  if length(ic) != length(initial_step)
    throw(ArgumentError("ic, $ic must be same length as initial_step
                        $initial_step"))
  end
  dim = length(ic)
  positions = Vector{Vector{promote_type(U,V)}}()
  for i ∈ 1:dim+1
    x = [ic[j] + ((j == i) ? initial_step[j] : zero(V)) for j ∈ 1:dim]
    push!(positions, x)
  end
  return Simplex(f, positions)
end
function Simplex(f::T, positions::U
    ) where {T<:Function, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}
  if length(unique(length.(positions))) != 1
    throw(ArgumentError("All entries in positions $positions must be the same
                        length"))
  end
  dim = length(positions) - 1
  vertex = Vertex(positions[1], f(positions[1]))
  vertices = Vector{typeof(vertex)}()
  push!(vertices, vertex)
  map(i->push!(vertices, Vertex(positions[i], f(positions[i]))), 2:dim+1)
  sort!(vertices, by=v->value(v))
  return Simplex(vertices)
end


import Base.length, Base.iterate, Base.push!, Base.iterate, Base.getindex
import Base.eachindex, Base.sort!
Base.length(s::Simplex) = length(s.vertices)
Base.push!(s::Simplex, v::Vertex) = push!(s.vertices, v)
Base.iterate(s::Simplex) = iterate(s.vertices)
Base.iterate(s::Simplex, counter) = iterate(s.vertices, counter)
Base.getindex(s::Simplex, index) = s.vertices[index]
Base.sort!(s::Simplex; kwargs...) = sort!(s.vertices; kwargs...)

dimensionality(s::Simplex) = length(s) - 1

remove!(s::Simplex, v::Vertex) = filter!(x -> !≡(x, v), s.vertices)

function getvertex(s::Simplex, i::Int)
  @assert 1 <= i <= length(s)
  @assert issorted(s)
  return s[i]
end

bestvertex(s::Simplex) = getvertex(s, 1)
worstvertex(s::Simplex) = getvertex(s, length(s))
secondworstvertex(s::Simplex) = getvertex(s, length(s)-1)

function findcentroid(f::T, s::Simplex) where {T<:Function}
  worst = worstvertex(s)
  g(v) = ≡(v, worst) ? zero(position(v)) : position(v)
  x = mapreduce(g, +, s) / (length(s) - 1)
  return Vertex(x, f(x))
end
function swapworst!(s::Simplex, forthis::Vertex)
  lengthbefore = length(s)
  remove!(s, worstvertex(s))
  push!(s, forthis)
  sort!(s, by=v->value(v))
  @assert length(s) == lengthbefore
  return nothing
end

function assessconvergence(simplex, xtol_abs, xtol_rel, ftol_abs, ftol_rel, stopval)
  value(bestvertex(simplex)) <= stopval && return :STOPVAL_REACHED

  toprocess = Set{Int}(1)
  processed = Set{Int}()
  while !isempty(toprocess)
    vi = pop!(toprocess)
    v = getvertex(simplex, vi)
    connectedto = Set{Int}()
    for (qi, q) ∈ enumerate(simplex)
      thisxtol = true
      for (i, (pv, pq)) ∈ enumerate(zip(position(v), position(q)))
        thisxtol &= isapprox(pv, pq, rtol=xtol_rel[i], atol=xtol_abs[i])
      end
      thisxtol && push!(connectedto, qi)
      thisxtol && for i in connectedto if i ∉ processed push!(toprocess, i) end end
    end
    push!(processed, vi)
  end
  allxtol = all(i ∈ processed for i ∈ 1:length(simplex))
  allxtol && return :XTOL_REACHED

  allftol = true
  for (vi, v) ∈ enumerate(simplex)
    for qi ∈ vi+1:length(simplex)
      q = getvertex(simplex, qi)
      allftol &= all(isapprox(value(v), value(q), rtol=ftol_rel, atol=ftol_abs))
    end
  end
  allftol && return :FTOL_REACHED

  return :CONTINUE
end

