struct Vertex{T, U}
  position::AbstractVector{T}
  value::U
end
value(v::Vertex) = v.value
position(v::Vertex) = v.position

# must explicitly use <= and >= because == can't overridden and will
# be used in conjunction with < to create a <=
import Base: isless, +, -, <=, >=, isequal, isnan
Base.:isless(a::Vertex, b::Vertex) = value(a) < value(b)
Base.:<=(a::Vertex, b::Vertex) = value(a) <= value(b)
Base.:>=(a::Vertex, b::Vertex) = value(a) >= value(b)
Base.:+(a::Vertex, b) = position(a) .+ b
Base.:-(a::Vertex, b::Vertex) = position(a) .- position(b)
function Base.:isequal(a::Vertex, b::Vertex)
  values_equal = value(a) == value(b) || (isnan(a) && isnan(b))
  positions_equal = all(position(a) .== position(b))
  return values_equal && positions_equal
end
Base.:isnan(a::Vertex) = isnan(value(a))
