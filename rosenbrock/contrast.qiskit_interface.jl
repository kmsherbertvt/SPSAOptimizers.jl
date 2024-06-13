# LOAD THE OPTIMIZATION ROUTINES
include("plugins/qiskit_interface.jl")
    import .QiskitInterfacePlugin: trajectory

# LOAD PLOTTING
import Plots
include("plotting/convergence.jl"); import .ConvergencePlots
include("plotting/cost.jl"); import .CostPlots
include("plotting/trajectory.jl"); import .TrajectoryPlots

# LOAD COST-FUNCTIONS
include("functions/powerquad.jl");
    import .PowerQuad
    PQ_FN = PowerQuad.lossfunction()
    PQ_GD = PowerQuad.gradient()
    PG_HS = PowerQuad.hessian()
include("functions/rosenbrock.jl");
    import .Rosenbrock
    RB_FN = Rosenbrock.lossfunction()
    RB_GD = Rosenbrock.gradient()
    RB_HS = Rosenbrock.hessian()

##########################################################################################
#= STANDARDIZE A TRAJECTORY/PLOT INTERFACE =#

curves = Dict{String,Any}()

function register!(label, L, trajectory_options, plot_options; plot=true)
    x0 = zeros(L)
    curves[label] = (
        label = label,
        L = L,
        trajectory_options = trajectory_options,
        plot_options = plot_options,
        plot = plot,
        python_RB = trajectory(:python, RB_FN, x0; trajectory_options...),
        python_PQ = trajectory(:python, PQ_FN, x0; trajectory_options...),
        julia_RB = trajectory(:julia, RB_FN, x0; trajectory_options...),
        julia_PQ = trajectory(:julia, PQ_FN, x0; trajectory_options...),
    )
end


##########################################################################################
#= REGISTER SOME CURVES =#

register!(
    "1SPSA", 2,
    (
        a0 = 0.1,
        A = 10.0,
        maxiter = 100,
        trust_region = true,
    ),
    (
        color = 1,
    )
)

register!(
    "2SPSA", 2,
    (
        a0 = 1.0,
        A = 10.0,
        maxiter = 100,
        second_order = true,
        blocking = true,
        allowed_increase = 0.0,
    ),
    (
        color = 2,
    )
)

register!(
    "2SPSA HI-D", 20,
    (
        a0 = 1.0,
        A = 10.0,
        maxiter = 100,
        second_order = true,
        blocking = true,
        allowed_increase = 0.0,
    ),
    (
        color = 3,
    )
)

register!(
    "2SPSA HI-D RES", 20,
    (
        a0 = 1.0,
        A = 10.0,
        maxiter = 100,
        second_order = true,
        blocking = true,
        allowed_increase = 0.0,
        resamplings = 20,
    ),
    (
        color = 4,
    )
)


##########################################################################################
#= PLOT THE DATA =#

DEFAULT_PLOT = (
    markerstrokewidth = 0,
    seriesalpha = 0.6,
    linewidth = 2,
)
PYTHON_PLOT = (
    markershape = :circle,
    linestyle = :dot,
)
JULIA_PLOT = (
    markershape = :star,
    linestyle = :solid,
)

function add_plots!(RB, PQ, Plotter, curve)
    Plotter.add!(RB, curve.python_RB;
        DEFAULT_PLOT..., PYTHON_PLOT...,
        label=curve.label,
        curve.plot_options...,
    )

    Plotter.add!(
        RB, curve.julia_RB;
        DEFAULT_PLOT..., JULIA_PLOT...,
        label=false,
        curve.plot_options...,
    )

    Plotter.add!(PQ, curve.python_PQ;
        DEFAULT_PLOT..., PYTHON_PLOT...,
        label=curve.label,
        curve.plot_options...,
    )

    Plotter.add!(
        PQ, curve.julia_PQ;
        DEFAULT_PLOT..., JULIA_PLOT...,
        label=false,
        curve.plot_options...,
    )
end

dir = "rosenbrock/fig"
prefix = "qiskit_interface"

cvg_RB = ConvergencePlots.init(; log=false, nfev=false)
cst_RB = CostPlots.init(; log=false)
trj_RB = TrajectoryPlots.init(RB_FN; )

cvg_PQ = ConvergencePlots.init(; log=true, nfev=false)
cst_PQ = CostPlots.init(; log=true)
trj_PQ = TrajectoryPlots.init(PQ_FN; )

for (label, curve) in pairs(curves)
    add_plots!(cvg_RB, cvg_PQ, ConvergencePlots, curve)
    add_plots!(cst_RB, cst_PQ, CostPlots, curve)
    add_plots!(trj_RB, trj_PQ, TrajectoryPlots, curve)
end

Plots.savefig(cvg_RB, "$dir/$prefix.RB.convergence.pdf")
Plots.savefig(cst_RB, "$dir/$prefix.RB.cost.pdf")
Plots.savefig(trj_RB, "$dir/$prefix.RB.trajectory.pdf")

Plots.savefig(cvg_PQ, "$dir/$prefix.PQ.convergence.pdf")
Plots.savefig(cst_PQ, "$dir/$prefix.PQ.cost.pdf")
Plots.savefig(trj_PQ, "$dir/$prefix.PQ.trajectory.pdf")