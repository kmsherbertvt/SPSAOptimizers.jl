#= Given a fixed budget M, and a fixed resamplings `n`,
    1SPSA can use K = M/2n while 2SPSA can use K = M/4n.
Which is better, and how does it change with n, and with the function?
=#

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

function do_1SPSA(FN, x0, K, n; trace=Trace())
    L = length(x0)
    # η = learning_rate(0.05, 0.602, K/10.0)
    η = learning_rate(0.1, 1.0, K/10.0)
    h = SPSAOptimizers.PowerSeries(0.1, 1/6)
    e = SPSAOptimizers.BernoulliDistribution(L=L, p=1.0)
    n_= SPSAOptimizers.IntDictStream(default=n)
    optimizer = SPSAOptimizers.SPSA1(L; η=η, h=h, e=e, n=n_)
    return SPSAOptimizers.optimize!(
        optimizer, FN.fn, x0;
        maxiter = K,
        trust = 1.0,
        tolerance = 0.0,
        trace = trace,
        tracefields = (:nfev, :fp, :xp, :g),
    )
end

function do_2SPSA(FN, x0, K, n; trace=Trace())
    L = length(x0)
    η = learning_rate(1.0, 1.0, K/10.0)
    # η = SPSAOptimizers.PowerSeries(120.0, 1.0, K/10.0)
    h = SPSAOptimizers.PowerSeries(0.1, 1/6)
    e = SPSAOptimizers.BernoulliDistribution(L=L, p=1.0)
    n_= SPSAOptimizers.IntDictStream(default=n)
    optimizer = SPSAOptimizers.SPSA2(L; η=η, h=h, e=e, n=n_)
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

data(FN, L, M, n, o) = (
    trace = Trace();
    if o == 1;
        do_1SPSA(FN, zeros(L), M÷2n, n; trace=trace);
    else;
        do_2SPSA(FN, zeros(L), M÷4n, n; trace=trace);
    end;
    data_from_trace(trace)
)

function register!(n, plot_options; plot=true)
    curves[n] = (
        label = "n=$n",
        n = n,
        plot_options = plot_options,
        plot = plot,
        RB_1 = data(RB, 20, 20000, n, 1),
        RB_2 = data(RB, 20, 20000, n, 2),
        PQ_1 = data(PQ, 20, 20000, n, 1),
        PQ_2 = data(PQ, 20, 20000, n, 2),
    )
end


##########################################################################################
#= REGISTER SOME CURVES =#

register!(1, (color = 1,))
register!(2, (color = 2,))
register!(4, (color = 3,))
register!(8, (color = 4,))
register!(16, (color = 5,))
register!(32, (color = 6,))
register!(64, (color = 7,))

##########################################################################################
#= PLOT THE DATA =#

DEFAULT_PLOT = (
    markerstrokewidth = 0,
    seriesalpha = 0.6,
    linewidth = 2,
)
PLOT_1 = (
    markershape = :circle,
    linestyle = :dot,
)
PLOT_2 = (
    markershape = :star,
    linestyle = :solid,
)

function add_plots!(RB, PQ, Plotter, curve)
    Plotter.add!(RB, curve.RB_1;
        DEFAULT_PLOT..., PLOT_1...,
        label=false,
        curve.plot_options...,
    )

    Plotter.add!(RB, curve.RB_2;
        DEFAULT_PLOT..., PLOT_2...,
        label=curve.label,
        curve.plot_options...,
    )

    Plotter.add!(PQ, curve.PQ_1;
        DEFAULT_PLOT..., PLOT_1...,
        label=false,
        curve.plot_options...,
    )

    Plotter.add!(PQ, curve.PQ_2;
        DEFAULT_PLOT..., PLOT_2...,
        label=curve.label,
        curve.plot_options...,
    )
end

dir = "rosenbrock/fig"
prefix = "12SPSA"

cvg_RB = ConvergencePlots.init(; log=false, nfev=false, ylims=[0.5,1.1], yticks=:auto)
cst_RB = CostPlots.init(; log=false, ylims=[0.5,1.1], yticks=:auto)
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