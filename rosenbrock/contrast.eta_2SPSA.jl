const Float = Float64
# LOAD THE OPTIMIZATION ROUTINES
import SPSAOptimizers
import SPSAOptimizers: Trace
include("plugins/spsa.jl")
    import .SPSAPlugin: data_from_trace

# LOAD PLOTTING
import Plots
include("plotting/convergence.jl"); import .ConvergencePlots
include("plotting/cost.jl"); import .CostPlots
include("plotting/trajectory.jl"); import .TrajectoryPlots

# LOAD COST-FUNCTIONS
include("functions/powerquad.jl");
    import .PowerQuad
    PQ = (
        fn = PowerQuad.lossfunction(),
        gd = PowerQuad.gradient(),
        hs = PowerQuad.hessian(),
    )
include("functions/rosenbrock.jl");
    import .Rosenbrock
    RB = (
        fn = Rosenbrock.lossfunction(),
        gd = Rosenbrock.gradient(),
        hs = Rosenbrock.hessian(),
    )



##########################################################################################
#= DEFINE EACH OF THE STRATEGIES =#

learning_rate(η0, α, A) = SPSAOptimizers.PowerSeries(η0 * (A+1)^α, α, A)

function do_2SPSA(FN, x0, K, η0; trace=Trace())
    L = length(x0)
    η = learning_rate(η0, 0.602, K/10.0)
    h = SPSAOptimizers.PowerSeries(0.1, 0.101)
    e = SPSAOptimizers.BernoulliDistribution(L=L, p=1.0)
    n = SPSAOptimizers.IntDictStream(default=2)
    optimizer = SPSAOptimizers.SPSA2(L; η=η, h=h, e=e, n=n)
    return SPSAOptimizers.optimize!(
        optimizer, FN.fn, x0;
        maxiter = K,
        # trust = 1.0,
        tolerance = 0.0,
        trace = trace,
        tracefields = (:nfev, :fp, :xp, :g),
    )
end


##########################################################################################
#= STANDARDIZE A TRAJECTORY/PLOT INTERFACE =#

curves = Dict{Float,Any}()

data(FN, L, K, η0) = (
    trace = Trace();
    do_2SPSA(FN, zeros(L), K, η0; trace=trace);
    data_from_trace(trace)
)

function register!(η0, plot_options; plot=true)
    curves[η0] = (
        label = "η0=$η0",
        η0 = η0,
        plot_options = plot_options,
        plot = plot,
        RB_l = data(RB,  2, 100, η0),
        RB_L = data(RB, 20, 1000, η0),
        PQ_l = data(PQ,  2, 100, η0),
        PQ_L = data(PQ, 20, 1000, η0),
    )
end


##########################################################################################
#= REGISTER SOME CURVES =#

register!(0.01, (color = 1,))
register!(0.05, (color = 2,))
register!(0.1, (color = 3,))
register!(0.5, (color = 4,))
register!(1.0, (color = 5,))

##########################################################################################
#= PLOT THE DATA =#

DEFAULT_PLOT = (
    markerstrokewidth = 0,
    seriesalpha = 0.6,
    linewidth = 2,
)
l_PLOT = (
    markershape = :circle,
    linestyle = :dot,
)
L_PLOT = (
    markershape = :star,
    linestyle = :solid,
)

function add_plots!(RB, PQ, Plotter, curve)
    Plotter.add!(RB, curve.RB_l;
        DEFAULT_PLOT..., l_PLOT...,
        label=curve.label,
        curve.plot_options...,
    )

    Plotter.add!(
        RB, curve.RB_L;
        DEFAULT_PLOT..., L_PLOT...,
        label=false,
        curve.plot_options...,
    )

    Plotter.add!(PQ, curve.PQ_l;
        DEFAULT_PLOT..., l_PLOT...,
        label=curve.label,
        curve.plot_options...,
    )

    Plotter.add!(
        PQ, curve.PQ_L;
        DEFAULT_PLOT..., L_PLOT...,
        label=false,
        curve.plot_options...,
    )
end

dir = "rosenbrock/fig"
prefix = "eta_2SPSA"

cvg_RB = ConvergencePlots.init(; log=false, nfev=false)
cst_RB = CostPlots.init(; log=false)
trj_RB = TrajectoryPlots.init(RB.fn; )

cvg_PQ = ConvergencePlots.init(; log=true, nfev=false)
cst_PQ = CostPlots.init(; log=true)
trj_PQ = TrajectoryPlots.init(PQ.fn; )

for label in sort(collect(keys(curves)))
    add_plots!(cvg_RB, cvg_PQ, ConvergencePlots, curves[label])
    add_plots!(cst_RB, cst_PQ, CostPlots, curves[label])
    add_plots!(trj_RB, trj_PQ, TrajectoryPlots, curves[label])
end

Plots.savefig(cvg_RB, "$dir/$prefix.RB.convergence.pdf")
Plots.savefig(cst_RB, "$dir/$prefix.RB.cost.pdf")
Plots.savefig(trj_RB, "$dir/$prefix.RB.trajectory.pdf")

Plots.savefig(cvg_PQ, "$dir/$prefix.PQ.convergence.pdf")
Plots.savefig(cst_PQ, "$dir/$prefix.PQ.cost.pdf")
Plots.savefig(trj_PQ, "$dir/$prefix.PQ.trajectory.pdf")