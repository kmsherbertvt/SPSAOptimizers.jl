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

x0 = zeros(2)

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

function calibrate_a0(x0, α, A; Δx=nothing)
    isnothing(Δx) && return (A+1)^α
    g0 = gd(x0)
    return (A+1)^α * Δx / LinearAlgebra.norm(g0)
end
function python_learningrate(x0, α, A; Δx=nothing)
    a0 = calibrate_a0(x0, α, A; Δx=Δx)
    return QiskitPlugin.wrapped_powerseries(a0, α, A)
end
function julia_learningrate(x0, α, A; Δx=nothing)
    a0 = calibrate_a0(x0, α, A; Δx=Δx)
    return (a0, α, A)
end

register!(
    "Qiskit 1SPSA",
    QiskitPlugin.trajectory(fn, x0;
        order = 1,
        maxiter = K,
        learning_rate = python_learningrate(x0, 0.602, K/10.0; Δx=0.2),
        perturbation = QiskitPlugin.wrapped_powerseries(0.1, 0.101),
        trust_region = true,
    );
    color = :green,
    # plot = false,
)

register!(
    "Qiskit 2SPSA",
    QiskitPlugin.trajectory(fn, x0;
        order = 2,
        maxiter = K,
        learning_rate = python_learningrate(x0, 0.602, K/10.0),
        perturbation = QiskitPlugin.wrapped_powerseries(0.1, 0.101),
        blocking = true,
    );
    color = :blue,
    # plot = false,
)

##########################################################################################

register!(
    "1SPSA",
    SPSAPlugin.trajectory(fn, x0;
        order = 1,
        maxiter = K,
        optimizeroptions = (
            η = julia_learningrate(x0, 0.602, K/10.0; Δx=0.2),
            h = (0.1, 0.101),
        ),
        trust = 1.0,
    );
    color = :red,
    # plot = false,
)

register!(
    "2SPSA",
    SPSAPlugin.trajectory(fn, x0;
        order = 2,
        maxiter = K,
        optimizeroptions = (
            η = (1.0, 0.602, K/10.0),
            h = (0.1, 0.101),
        ),
        tolerance = 0.0,
    );
    color = :purple,
    # plot = false,
)

##########################################################################################




# RUN THE SPSA OPTIMIZATION


##########################################################################################

dir = "rosenbrock/fig"
prefix = "methods"

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