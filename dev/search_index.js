var documenterSearchIndex = {"docs":
[{"location":"#NelderMead.jl-Documentation","page":"NelderMead.jl Documentation","title":"NelderMead.jl Documentation","text":"","category":"section"},{"location":"","page":"NelderMead.jl Documentation","title":"NelderMead.jl Documentation","text":"CurrentModule = NelderMead","category":"page"},{"location":"","page":"NelderMead.jl Documentation","title":"NelderMead.jl Documentation","text":"Modules = [NelderMead]","category":"page"},{"location":"#NelderMead.optimise!-Union{Tuple{U}, Tuple{T}, Tuple{F}, Tuple{NelderMead.Simplex{T, U}, F}} where {F<:Function, T<:Real, U}","page":"NelderMead.jl Documentation","title":"NelderMead.optimise!","text":"optimise!(s, f; kwargs...)\n\nFind minimum of function, f, starting from Simplex, s, with options passed in via kwargs.\n\nKeyword Arguments\n\nstopval (default sqrt(eps())): stopping criterion when function evaluates\n\nequal to or less than stopval\n\nxtol_abs (default zeros(T)) .* ones(Bool, dimensionality(s)): stop if\n\nthe vertices of simplex get within this absolute tolerance\n\nxtol_rel (default eps(T)) .* ones(Bool, dimensionality(s)): stop if\n\nthe vertices of simplex get within this relative tolerance\n\nftol_abs (default zero(real(U))): stop if function evaluations at the\n\nvertices are close to one another by this absolute tolerance\n\nftol_rel (default 1000eps(real(U))): stop if function evaluations at the\n\nvertices are close to one another by this relative tolerance\n\nmaxiters (default 1000): maximum number of iterations of the Nelder Mead\n\nalgorithm\n\ntimelimit (default Inf): stop if it takes longer than this in seconds\nα (default 1): Reflection factor\nβ (default 0.5): Contraction factor\nγ (default 2): Expansion factor\nδ (default 0.5): Shrinkage factor\n\n\n\n\n\n","category":"method"},{"location":"#NelderMead.optimise-Union{Tuple{T}, Tuple{F}, Tuple{F, AbstractVector{T}, AbstractVector{T}, AbstractVector{var\"#s3\"} where var\"#s3\"<:Integer}} where {F<:Function, T<:Number}","page":"NelderMead.jl Documentation","title":"NelderMead.optimise","text":"optimise(f, lower, upper, gridsize; kwargs...)\n\nFind minimum of function, f, first creating a 2 .* prod(gridsize) Simplices within a hyper-rectangle spanning from lower to upper, and options passed in via kwargs.\n\n\n\n\n\n","category":"method"},{"location":"#NelderMead.optimise-Union{Tuple{U}, Tuple{V}, Tuple{W}, Tuple{T}, Tuple{T, U}} where {T<:Function, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}","page":"NelderMead.jl Documentation","title":"NelderMead.optimise","text":"optimise(f, initial_vertex_positions; kwargs...)\n\nFind minimum of function, f, first creating a Simplex from vertices at initial_vertex_positions, and options passed in via kwargs.\n\n\n\n\n\n","category":"method"},{"location":"#NelderMead.optimise-Union{Tuple{V}, Tuple{U}, Tuple{T}, Tuple{T, AbstractVector{U}, AbstractVector{V}}} where {T<:Function, U<:Number, V<:Number}","page":"NelderMead.jl Documentation","title":"NelderMead.optimise","text":"optimise(f, initial_positions, initial_step; kwargs...)\n\nFind minimum of function, f, first creating a Simplex using a starting vertex position, initial_position, and other vertices initial_step away from that point in all directions, and options passed in via kwargs.\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"NelderMead.jl Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"NelderMead.jl Documentation","title":"NelderMead.jl Documentation","text":"","category":"page"}]
}
