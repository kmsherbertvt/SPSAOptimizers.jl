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
include("functions/rosenbrock.jl"); import .Rosenbrock
fn = Rosenbrock.lossfunction()
gd = Rosenbrock.gradient()
hs = Rosenbrock.hessian()

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

Ls = 2#[2, 3, 5, 8, 13, 21]
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
            trust_region = true,
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
        );
        color = i,
        seriesalpha = 0.6,
        linestyle = :dash,
        markershape = :star,
        label = false,
        # plot = false,
    )

    register!(
        "BFGS L=$L",
        OptimPlugin.trajectory(fn, x0; order=1.5);
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
prefix = "dimensions"

# MAKE THE CONVERGENCE PLOTS
convergence = ConvergencePlots.init(; log=false, nfev=false)
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




#=

TODO:

Okay, we've learned a lot.

Resampling is super important for 2SPSA to work. Like Spall said, sorta under his breath. :P

The blocking strat to mimic large resampling in the beginning doesn't work super well because (I think..?) with η=1.0, the points are just so far apart and the finite differences don't reliably average, or perhaps the gradients themselves are so erratic over that dispersed population.

- I think you'd best set resamplings to be a stream after all.
- Also I think this is suggestive that your p² finite differencing might be worthwhile.

    BUT don't forget the primary goal here is to emulate qiskit. Unfortunately, despite documentation, resamplings does NOT accept a generator! It does however accept a dict so I guess it's still doable. Methinks we should have a Dict-backed Stream (serializes the k->n dict, plus a default).

So, I think it's time to finish your second-order implementation,
    complete with optimize! utilities.

    As to `trust_region`...
    ...I think it's really conceptually important for all of the state changes to happen within a single iterate!. That includes generating the next η, and that happens after the trust_region business, so trust_region must indeed be where it is. A bit unfortunate but there it is.

    optimize! probably SHOULD dispatch differently on the different orders, since the kwarg interface is different, with the Hessian and whatnot. Clean up the whole interface. Having p separate from g is good I htink.

    Remember, we need:
    - tracing enablable for nfev, x, f, g, and H. p too I guess (remember p does NOT include η).
    - blocking with given tolerance
    - averaging over the last so many points - HEY! this average includes rejected points? NO. The loop is continued before `last_steps` is updated, so `last_avg` is the avg of the last so many ACCEPTED steps.
    - hessian_delay
=#