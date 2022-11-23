![CI](https://github.com/jwscook/NelderMead.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/jwscook/NelderMead.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jwscook/NelderMead.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://jwscook.github.io/NelderMead.jl/dev/)

# NelderMead.jl

Pure Julia Nelder Mead optimisation

```julia
using NelderMead, CairoMakie

xs, ys, zs, vs = Float64[], Float64[], Float64[], Float64[]

function rosenbrock(x)
  output = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
  output += (1 - x[2])^2 + 100 * (x[3] - x[2]^2)^2
  push!(xs, x[1]); push!(ys, x[2]); push!(zs, x[3]); push!(vs, output)
  return output
end

result = NelderMead.optimise(rosenbrock, zeros(3), ones(3) ./ 10)
position, minvalue, returncode, iters, simplex = result

fig, ax, scat = scatter(0 .* xs .+ 1.2, ys, zs, color=:grey, alpha=0.5,
                        axis=(;type=Axis3))
scatter!(xs, 0 .* ys .+ 1.2, zs, color=:grey, alpha=0.5)
scatter!(xs, ys, 0 .* zs .- 0.2, color=:grey, alpha=0.5)
scatter!(xs, ys, zs, color=log10.(vs))
limits!(ax, -0.2, 1.2, -0.2, 1.2, -0.2, 1.2)

save("Rosenbrock3D.png", fig)
```

![NelderMead jl](https://user-images.githubusercontent.com/15519866/197048856-0f94766d-93ef-41ba-a8d8-b0b3b1e67854.png)

