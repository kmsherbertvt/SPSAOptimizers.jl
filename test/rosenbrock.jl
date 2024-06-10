#=  =#

import Plots
import Optim, LineSearches
import SPSAOptimizers

function rosenbrock(a=1.0, b=100.0)
    return x -> (
        (x1, x2) = x;
        (a - x1)^2 + b * (x2 - x1^2)^2
    )
end




function plot_fn(fn; xlims=[-0.5,1.5], ylims=[-0.5,1.5])
    x = xlims[1]:.1:xlims[2]
    y = ylims[1]:.1:ylims[2]
    return Plots.contourf(
        x, y, (x, y) -> fn([x,y]);
        levels=[0, 10.0 .^ (-4:1:4)...],
        # framestyle=:origin,
    )
end

function plot_trajectory!(plt, coords; kwargs...)
    Plots.plot!(plt,
        coords[:,1], coords[:,2];
        lw = 3,
        shape=:circle,
        label = false,
        kwargs...
    )
end











fn = rosenbrock()
x0 = zeros(2)





############################################################
#= Gradient Descent, but let's call it FDSA =#
import Optim, LineSearches
bfgs = Optim.GradientDescent()
options = Optim.Options(
    store_trace = true,
    extended_trace = true,
)
result = Optim.optimize(fn, x0, bfgs, options)
fdsa_f = [step.value for step in Optim.trace(result)]
fdsa_x = [step.metadata["x"] for step in Optim.trace(result)]


############################################################
#= SPSA =#
# TODO: Qiskit implementation as a better control curve.
η = SPSAOptimizers.PowerSeries(a0=0.2, γ=0.619)
h = SPSAOptimizers.PowerSeries(a0=0.2, γ=0.619)
e = SPSAOptimizers.BernoulliDistribution(L=2)
spsa = SPSAOptimizers.SPSA(
    η=η,
    h=h,
    e=e,
    trust_region=1.0,
)
niter = 1000
spsa_f = Float64[fn(x0)]
spsa_x = Vector{Float64}[x0]
f = Ref(0.0)
x = deepcopy(x0)
for iter in 1:niter
    SPSAOptimizers.Optimizers.iterate!(spsa, x, fn; f=f)
    push!(spsa_f, f[])
    push!(spsa_x, deepcopy(x))
end



############################################################
#= Plot 'em. =#

pltf = Plots.plot(;
    ylabel = "Loss Function",
    yscale = :log10,
    xlabel = "Iteration"
)
Plots.plot!(pltf,
    fdsa_f;
    lw=3,
    label="FDSA", color=:yellow, alpha=0.8,
)
Plots.plot!(pltf,
    spsa_f;
    lw=3,
    label="SPSA", color=:red, alpha=0.8,
)


pltx = plot_fn(fn)
plot_trajectory!(pltx,
    transpose(reduce(hcat, fdsa_x));
    label="FDSA", color=:yellow, alpha=0.8,
)
plot_trajectory!(pltx,
    transpose(reduce(hcat, spsa_x));
    label="SPSA", color=:red, alpha=0.8,
)