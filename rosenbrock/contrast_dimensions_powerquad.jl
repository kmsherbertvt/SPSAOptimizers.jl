# LOAD ALL OPTIMIZATION ROUTINES
include("plugins/qiskit.jl"); import .QiskitPlugin
include("plugins/optim.jl"); import .OptimPlugin
include("plugins/spsa.jl"); import .SPSAPlugin

# LOAD PLOTTING
import Plots
include("plotting/convergence.jl"); import .ConvergencePlots
include("plotting/cost.jl"); import .CostPlots
include("plotting/trajectory.jl"); import .TrajectoryPlots

# LOAD COST-FUNCTIONS
include("functions/powerquad.jl"); import .PowerQuad
fn = PowerQuad.lossfunction()
gd = PowerQuad.gradient()
hs = PowerQuad.hessian()

import LinearAlgebra

##########################################################################################

curves = Dict{String,Any}()
register!(label, data; plot=true, kwargs...) = (
    curves[label] = (data=data, label=label, plot=plot, kwargs=kwargs)
)

##########################################################################################

# register!(
#     "Gradient Descent",
#     OptimPlugin.trajectory(fn, x0; order=1);
#     color = :bronze,
#     plot = false,
# )

# register!(
#     "BFGS",
#     OptimPlugin.trajectory(fn, x0; order=1.5);
#     color = :gold,
#     # plot = false,
# )

# register!(
#     "Newton",
#     OptimPlugin.trajectory(fn, x0; order=2);
#     color = :silver,
#     # plot = false,
# )

##########################################################################################

K = 100

function calibrated_generator(x0, α, A; Δx=nothing)
    # If Δx is not provided, default to η[0]=1.0.
    isnothing(Δx) && return QiskitPlugin.wrapped_powerseries((A+1)^α, α, A)

    # Otherwise, follow Spall's calibration protocol, implemented in qiskit.
    g0 = gd(x0);
    QiskitPlugin.wrapped_powerseries(Δx/LinearAlgebra.norm(g0) * (A+1)^α, α, A)
end
h = QiskitPlugin.wrapped_powerseries(0.1, 0.101)

#= Realization:

For order=2,
    Δx ≠ η * g0
    Δx = η * Hk * g0

Moreover, for order=2, you really want η to be 1.0, don't you? Target magnitude be damned!

=#

Ls = [2, 3, 5, 8, 13, 21]
for (i, L) in enumerate(Ls)

    x0 = zeros(L)
    η1 = calibrated_generator(x0, 0.602, K/10.0; Δx=(√L)/10.0)
    η2 = calibrated_generator(x0, 0.602, K/10.0)

    register!(
        "1SPSA L=$L",
        QiskitPlugin.trajectory(fn, x0;
            order = 1,
            maxiter = K,
            learning_rate = η1,
            perturbation = h,
            # trust_region = true,
        );
        color = i,
        seriesalpha = 0.6,
        linestyle = :solid,
        markershape = :circle,
        label = "L=$L",
        # plot = false,
    )

    register!(
        "2SPSA L=$L",
        QiskitPlugin.trajectory(fn, x0;
            order = 2,
            maxiter = K,
            learning_rate = η2,
            perturbation = h,
            blocking = true,
            resamplings = L,
        );
        color = i,
        seriesalpha = 0.6,
        linestyle = :dash,
        markershape = :star,
        label = false,
        # plot = false,
    )

    register!(
        "Newton L=$L",
        OptimPlugin.trajectory(fn, x0; order=2);
        color = i,
        seriesalpha = 0.6,
        linestyle = :dot,
        markershape = :square,
        label = false,
        # plot = false,
    )
end

##########################################################################################

dir = "rosenbrock/fig"
prefix = "dimensions_powerquad"

# MAKE THE CONVERGENCE PLOTS
convergence = ConvergencePlots.init(; log=true, nfev=false)
for (label, curve) in pairs(curves)
    curve.plot || continue
    ConvergencePlots.add!(convergence, curve.data; label=label, curve.kwargs...)
end
Plots.savefig(convergence, "$dir/$prefix.convergence.pdf")

# MAKE THE COST PLOTS
cost = CostPlots.init(; log=true)
for (label, curve) in pairs(curves)
    curve.plot || continue
    CostPlots.add!(cost, curve.data; label=label, curve.kwargs...)
end
Plots.savefig(cost, "$dir/$prefix.cost.pdf")

# MAKE THE COST PLOTS
trajectory = TrajectoryPlots.init(fn; )
for (label, curve) in pairs(curves)
    curve.plot || continue
    TrajectoryPlots.add!(trajectory, curve.data; label=label, curve.kwargs...)
end
Plots.savefig(trajectory, "$dir/$prefix.trajectory.pdf")