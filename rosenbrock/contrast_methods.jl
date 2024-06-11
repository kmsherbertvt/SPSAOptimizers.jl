# LOAD ALL OPTIMIZATION ROUTINES
include("plugins/qiskit.jl"); import .QiskitPlugin
include("plugins/optim.jl"); import .OptimPlugin
include("plugins/spsa.jl"); import .SPSAPlugin

# LOAD PLOTTING
import Plots
include("plotting/convergence.jl"); import .ConvergencePlots
include("plotting/cost.jl"); import .CostPlots
include("plotting/trajectory.jl"); import .TrajectoryPlots

##########################################################################################

# PREP THE SYSTEM
function rosenbrock(a=1.0, b=100.0)
    return params -> (
        (x, y) = params;
        (a - x)^2 + b * (y - x^2)^2
    )
end

function rosenbrock_gradient(a=1.0, b=100.0)
    return params -> (
        (x, y) = params;
        [
            -2(a - x) - 4b * (y - x^2) * x,
            2b * (y - x^2),
        ]
    )
end

function rosenbrock_hessian(a=1.0, b=100.0)
    return params -> (
        (x, y) = params;
        [
            2 - 4b * (y - 3x^2)         -4b*x
            -4b * x                     2b
        ]
    )
end

fn = rosenbrock()
x0 = zeros(2)

gd = rosenbrock_gradient()
hs = rosenbrock_hessian()

##########################################################################################

# RUN THE OPTIM OPTIMIZATION

# optim = OptimPlugin.trajectory(fn, x0)
# bfgs = OptimPlugin.trajectory(fn, x0; order=:bfgs)
# newton = OptimPlugin.trajectory(fn, x0; order=2)

# RUN THE QISKIT OPTIMIZATION

η, h = QiskitPlugin.calibrate(
    fn, x0,
    target_magnitude = 0.2,
    alpha = 0.101,
)

# begin
#     eta = η(); eps = h();
#     plt = Plots.plot();
#     Plots.plot!(plt, [eta.__next__() for i in 1:200], label="eta")
#     Plots.plot!(plt, [eps.__next__() for i in 1:200], label="eps")
#     Plots.gui(plt)
# end



qiskit_1 = QiskitPlugin.trajectory(fn, x0;
    learning_rate = η,
    perturbation = h,
    trust_region = true,
)

qiskit_2 = QiskitPlugin.trajectory(fn, x0;
    order = 2,
    learning_rate = η,
    perturbation = h,
    # blocking=true,
    # resamplings = 4,
    # trust_region = true,
    # hessian_delay = 4,
    # initial_hessian = nothing, # TODO
)

xr = [last(qiskit_1.x), last(qiskit_1.y)]
ηr, hr = QiskitPlugin.calibrate(
    fn, xr,
    target_magnitude = 0.2,
    alpha = 0.101,
)
qiskit_r = QiskitPlugin.trajectory(fn, xr;
    order = 2,
    learning_rate = ηr,
    perturbation = hr,
    # blocking=true,
    # resamplings = 2,
    # initial_hessian = qiskit_2.H,
)

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


η0 = η().__next__()
h0 = h().__next__()

ηr0 = ηr().__next__()
hr0 = hr().__next__()

# RUN THE SPSA OPTIMIZATION

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

fig_dir = "rosenbrock/fig"

# MAKE THE CONVERGENCE PLOTS
convergence = ConvergencePlots.init(; log=false, nfev=false)
# ConvergencePlots.add!(convergence, bfgs; color=:cyan, label="BFGS")
# ConvergencePlots.add!(convergence, newton; color=:orange, label="Newton")
# ConvergencePlots.add!(convergence, spsa_1; color=:red, label="SPSA (1)")
ConvergencePlots.add!(convergence, qiskit_1; color=:purple, label="Qiskit (1)")
ConvergencePlots.add!(convergence, qiskit_2; color=:gold, label="Qiskit (2)")
ConvergencePlots.add!(convergence, qiskit_r; color=:silver, label="Qiskit (r)")
Plots.savefig(convergence, "$fig_dir/methods.convergence.pdf")

# MAKE THE COST PLOTS
cost = CostPlots.init(; log=false)
# CostPlots.add!(cost, bfgs; color=:cyan, label="BFGS")
# CostPlots.add!(cost, newton; color=:orange, label="Newton")
# CostPlots.add!(cost, spsa_1; color=:red, label="SPSA (1)")
CostPlots.add!(cost, qiskit_1; color=:purple, label="Qiskit (1)")
CostPlots.add!(cost, qiskit_2; color=:gold, label="Qiskit (2)")
CostPlots.add!(cost, qiskit_r; color=:silver, label="Qiskit (r)")
Plots.savefig(cost, "$fig_dir/methods.cost.pdf")

# MAKE THE COST PLOTS
trajectory = TrajectoryPlots.init(fn;
    # xlims=(min(minimum(qiskit.x),minimum(spsa.x)),max(maximum(qiskit.x),maximum(spsa.x))),
    # ylims=(min(minimum(qiskit.y),minimum(spsa.y)),max(maximum(qiskit.y),maximum(spsa.y))),
)
# TrajectoryPlots.add!(trajectory, bfgs; color=:cyan, label="BFGS")
# TrajectoryPlots.add!(trajectory, newton; color=:orange, label="Newton")
# TrajectoryPlots.add!(trajectory, spsa_1; color=:red, label="SPSA (1)")
TrajectoryPlots.add!(trajectory, qiskit_1; color=:purple, label="Qiskit (1)")
TrajectoryPlots.add!(trajectory, qiskit_2; color=:gold, label="Qiskit (2)")
TrajectoryPlots.add!(trajectory, qiskit_r; color=:silver, label="Qiskit (r)")
Plots.savefig(trajectory, "$fig_dir/methods.trajectory.pdf")