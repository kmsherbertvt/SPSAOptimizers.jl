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

function calibrated_generator(x0, α, A; Δx=nothing)
    # If Δx is not provided, default to η[0]=1.0.
    isnothing(Δx) && return QiskitPlugin.wrapped_powerseries((A+1)^α, α, A)

    # Otherwise, follow Spall's calibration protocol, implemented in qiskit.
    g0 = gd(x0);
    QiskitPlugin.wrapped_powerseries(Δx/LinearAlgebra.norm(g0) * (A+1)^α, α, A)
end
η1 = calibrated_generator(x0, 0.602, K/10.0)
η2 = calibrated_generator(x0, 0.602, K/10.0)
h = QiskitPlugin.wrapped_powerseries(0.1, 0.101)

register!(
    "Qiskit 1SPSA",
    QiskitPlugin.trajectory(fn, x0;
        # order = 2,
        maxiter = K,
        learning_rate = η1,
        perturbation = h,
        trust_region = true,
        # hessian_delay = K,  # so, trajectory is that of order=1
    );
    color = :green,
    # plot = false,
)

xr1 = curves["Qiskit 1SPSA"].data.x[end,:]
ηr1 = calibrated_generator(xr1, 0.602, K/10.0)
# Hr1 = curves["Qiskit 1SPSA"].data.optimizer._smoothed_hessian

register!(
    "Qiskit 2SPSA",
    QiskitPlugin.trajectory(fn, x0;
        order = 2,
        maxiter = K,
        learning_rate = η2,
        perturbation = h,
        blocking = true,
    );
    color = :blue,
    # plot = false,
)

xr2 = curves["Qiskit 2SPSA"].data.x[end,:]
ηr2 = calibrated_generator(xr2, 0.602, K/10.0)
Hr2 = curves["Qiskit 2SPSA"].data.optimizer._smoothed_hessian

##########################################################################################

register!(
    "Qiskit 2SPSA (From 1SPSA)",
    QiskitPlugin.trajectory(fn, xr1;
        order = 2,
        maxiter = K,
        learning_rate = ηr1,
        perturbation = h,
        # initial_hessian = Hr1,
        blocking = true,
    );
    color = :teal,
    # plot = false,
)

register!(
    "Qiskit 2SPSA (From 2SPSA)",
    QiskitPlugin.trajectory(fn, xr2;
        order = 2,
        maxiter = K,
        learning_rate = ηr2,
        perturbation = h,
        initial_hessian = Hr2,
        blocking = true,
    );
    color = :cyan,
    # plot = false,
)

##########################################################################################

# spsa_1 = SPSAPlugin.trajectory(fn, x0;
#     η = (η0, 0.602),
#     h = (h0, 0.101),
#     p = 3,
#     trust_region=1.0,
# )

# spsa_2 = SPSAPlugin.trajectory(fn, x0;
#     order = 2,
#     η = (η0, 0.602),
#     h = (h0, 0.101),
#     p = 3,
#     trust_region=1.0,
# )

##########################################################################################

#=

Huh.

There's an interesting catch-22.
Applying the trust_region with second_order is just dumb;
    maybe you refine direction a little bit,
    but the magnitude refinement is always entirely ignored,
    so convergence is zilch.
Without the trust_region, it ... starts happily going the wrong way..?
Presumably because the single-sample Hessian is not very accurate.
You can try to turn on the delay...
    ...but then you instantly remember why we use the trust_region. :)

I think we should see what happens if we provide a more robustly sampled initial_hessian.
However, simply increasing `resamplings` should _do_ that.
    (Which is a point: an iteration dependent `resamplings` has that utility.)
But this results in a trajectory converging as slowly as with trust_region. Why..? :(

THINGS TO TRY
- Start 2SPSA from the final point of SPSA?

  For that matter, it may be worth resetting eta and h (but not H) after the first 100 steps of 2SPSA (ie. run again but using initial_hessian = qiskit_2._smoothed_hessian.

        Sure it helped. But it's no cure.

- Try single-sample no trust_region but with blocking?

        Helped. No cure.

  Aye, it'll just keep refining the Hessian 'til it gives the right direction. But I'm afraid eta is still decaying too fast. :/ Could we please actually PLOT these default series, so we have some better intuition for them?

        Yeah, eta dies off way too fast. eps is fine. Kinda already knew that...

- h is supposed to be an order of magnitude larger? Don't think it can go over 1.0, but it's something to try...

        Did not help. :)

- Spall does say more resamplings should help "especially in the presence of noise".

        Good for him. The smoothed Hessian for qiskit_2 actually does seem to be reasonably accurate here. Although, admittedly, the inverse Hessian not so much.



Actually very good luck starting from (first-order) SPSA. Much better improvement than any of the other things, though doing all of them is "best".

I think the next step is nothing more nor less than reading the papers to understand what Spall thinks a good power series is. Clearly qiskit's default alpha is worthless, but in what context did they select it?

=#



# RUN THE SPSA OPTIMIZATION


##########################################################################################

fig_dir = "rosenbrock/fig"

# MAKE THE CONVERGENCE PLOTS
convergence = ConvergencePlots.init(; log=true, nfev=false)
for (label, curve) in pairs(curves)
    curve.plot || continue
    ConvergencePlots.add!(convergence, curve.data; label=label, curve.kwargs...)
end
Plots.savefig(convergence, "$fig_dir/methods.convergence.pdf")

# MAKE THE COST PLOTS
cost = CostPlots.init(; log=true)
for (label, curve) in pairs(curves)
    curve.plot || continue
    CostPlots.add!(cost, curve.data; label=label, curve.kwargs...)
end
Plots.savefig(cost, "$fig_dir/methods.cost.pdf")

# MAKE THE COST PLOTS
trajectory = TrajectoryPlots.init(fn; )
for (label, curve) in pairs(curves)
    curve.plot || continue
    TrajectoryPlots.add!(trajectory, curve.data; label=label, curve.kwargs...)
end
Plots.savefig(trajectory, "$fig_dir/methods.trajectory.pdf")