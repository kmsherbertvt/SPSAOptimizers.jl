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

fn = rosenbrock()
x0 = zeros(2)

##########################################################################################

# RUN THE OPTIM OPTIMIZATION

optim = OptimPlugin.trajectory(fn, x0)
bfgs = OptimPlugin.trajectory(fn, x0; order=:bfgs)
newton = OptimPlugin.trajectory(fn, x0; order=2)

# RUN THE QISKIT OPTIMIZATION

η, h = QiskitPlugin.calibrate(fn, x0, target_magnitude=0.2)

qiskit_1 = QiskitPlugin.trajectory(fn, x0;
    learning_rate = η,
    perturbation = h,
    trust_region = true,
)

qiskit_2 = QiskitPlugin.trajectory(fn, x0;
    order = 2,
    learning_rate = η,
    perturbation = h,
    resamplings = 25,
    # trust_region = true,
    # hessian_delay = 4,
    # initial_hessian = nothing, # TODO
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

=#


η0 = η().__next__()
h0 = h().__next__()

# RUN THE SPSA OPTIMIZATION

spsa_1 = SPSAPlugin.trajectory(fn, x0;
    η = (η0, 0.602),
    h = (h0, 0.101),
    p = 3,
    trust_region=1.0,
)

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
ConvergencePlots.add!(convergence, bfgs; color=:cyan, label="BFGS")
ConvergencePlots.add!(convergence, newton; color=:orange, label="Newton")
ConvergencePlots.add!(convergence, spsa_1; color=:red, label="SPSA (1)")
ConvergencePlots.add!(convergence, qiskit_1; color=:purple, label="Qiskit (1)")
ConvergencePlots.add!(convergence, qiskit_2; color=:gold, label="Qiskit (2)")
Plots.savefig(convergence, "$fig_dir/methods.convergence.pdf")

# MAKE THE COST PLOTS
cost = CostPlots.init(; log=false)
CostPlots.add!(cost, bfgs; color=:cyan, label="BFGS")
CostPlots.add!(cost, newton; color=:orange, label="Newton")
CostPlots.add!(cost, spsa_1; color=:red, label="SPSA (1)")
CostPlots.add!(cost, qiskit_1; color=:purple, label="Qiskit (1)")
CostPlots.add!(cost, qiskit_2; color=:gold, label="Qiskit (2)")
Plots.savefig(cost, "$fig_dir/methods.cost.pdf")

# MAKE THE COST PLOTS
trajectory = TrajectoryPlots.init(fn;
    # xlims=(min(minimum(qiskit.x),minimum(spsa.x)),max(maximum(qiskit.x),maximum(spsa.x))),
    # ylims=(min(minimum(qiskit.y),minimum(spsa.y)),max(maximum(qiskit.y),maximum(spsa.y))),
)
TrajectoryPlots.add!(trajectory, bfgs; color=:cyan, label="BFGS")
TrajectoryPlots.add!(trajectory, newton; color=:orange, label="Newton")
TrajectoryPlots.add!(trajectory, spsa_1; color=:red, label="SPSA (1)")
TrajectoryPlots.add!(trajectory, qiskit_1; color=:purple, label="Qiskit (1)")
TrajectoryPlots.add!(trajectory, qiskit_2; color=:gold, label="Qiskit (2)")
Plots.savefig(trajectory, "$fig_dir/methods.trajectory.pdf")