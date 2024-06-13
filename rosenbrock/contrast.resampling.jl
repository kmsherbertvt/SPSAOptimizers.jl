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

function best_1SPSA(FN, x0, K; trace=Trace())
    L = length(x0)
    η = learning_rate(0.05, 0.602, K/10.0)
    h = SPSAOptimizers.PowerSeries(0.1, 0.101)
    n = SPSAOptimizers.IntDictStream(default=2)
    optimizer = SPSAOptimizers.SPSA1(L; η=η, h=h, n=n)
    return SPSAOptimizers.optimize!(
        optimizer, FN.fn, x0;
        maxiter = K,
        trust = 1.0,
        trace = trace,
        tracefields = (:nfev, :fp, :xp, :g),
    )
end

function LL_strategy(FN, x0, K; trace=Trace())
    L = length(x0)
    η = learning_rate(1.0, 0.602, K/10.0)
    h = SPSAOptimizers.PowerSeries(0.1, 0.101)
    n = SPSAOptimizers.IntDictStream(dict=Dict(k=>L for k in 1:L), default=1)
    optimizer = SPSAOptimizers.SPSA2(L; η=η, h=h, n=n)
    return SPSAOptimizers.optimize!(
        optimizer, FN.fn, x0;
        maxiter = K,
        tolerance = 0.0,
        trace = trace,
        tracefields = (:nfev, :fp, :xp, :g),
    )
end

function H0_strategy(FN, x0, K; trace=Trace())
    L = length(x0)
    η = learning_rate(1.0, 0.602, K/10.0)
    h = SPSAOptimizers.PowerSeries(0.1, 0.101)
    H = SPSAOptimizers.TrajectoryHessian(L); H.H .= FN.hs(x0); H.k[] = L^2
    optimizer = SPSAOptimizers.SPSA2(L; H=H, η=η, h=h)
    return SPSAOptimizers.optimize!(
        optimizer, FN.fn, x0;
        maxiter = K,
        tolerance = 0.0,
        trace = trace,
        tracefields = (:nfev, :fp, :xp, :g),
    )
end

function restart_LL_strategy(FN, x0, K; trace=Trace())
    record = LL_strategy(FN, x0, K; trace=trace)
    return LL_strategy(FN, record.xp, K; trace=trace)
end

function restart_H0_strategy(FN, x0, K; trace=Trace())
    record = H0_strategy(FN, x0, K; trace=trace)
    return H0_strategy(FN, record.xp, K; trace=trace)
end

function descent_LL_strategy(FN, x0, K; trace=Trace())
    record = best_1SPSA(FN, x0, K; trace=trace)
    return LL_strategy(FN, record.xp, K; trace=trace)
end

function descent_H0_strategy(FN, x0, K; trace=Trace())
    record = best_1SPSA(FN, x0, K; trace=trace)
    return H0_strategy(FN, record.xp, K; trace=trace)
end

function warmup_strategy(FN, x0, K; trace=Trace())
    L = length(x0)
    η = learning_rate(1.0, 0.602, K/10.0)
    h = SPSAOptimizers.PowerSeries(0.1, 0.101)
    warmup_optimizer = SPSAOptimizers.SPSA2(L; η=η, h=h)
    record = SPSAOptimizers.optimize!(
        warmup_optimizer, FN.fn, x0;
        maxiter = K,
        warmup = K,
        trust = 1.0,
        trace = trace,
        tracefields = (:nfev, :fp, :xp, :g),
    )

    η = learning_rate(1.0, 0.602, K/10.0)
    h = SPSAOptimizers.PowerSeries(0.1, 0.101)
    H = deepcopy(warmup_optimizer.H)
    optimizer = SPSAOptimizers.SPSA2(L; H=H, η=η, h=h)
    return SPSAOptimizers.optimize!(
        optimizer, FN.fn, x0;
        maxiter = K,
        tolerance = 0.0,
        trace = trace,
        tracefields = (:nfev, :fp, :xp, :g),
    )
end


##########################################################################################
#= STANDARDIZE A TRAJECTORY/PLOT INTERFACE =#

curves = Dict{String,Any}()

data(FN, L, K, strategy) = (
    trace = Trace();
    strategy(FN, zeros(L), K; trace=trace);
    data_from_trace(trace)
)

function register!(label, strategy, plot_options; plot=true)
    curves[label] = (
        label = label,
        strategy = strategy,
        plot_options = plot_options,
        plot = plot,
        RB_l = data(RB,  2, 100, strategy),
        RB_L = data(RB, 20, 1000, strategy),
        PQ_l = data(PQ,  2, 100, strategy),
        PQ_L = data(PQ, 20, 1000, strategy),
    )
end


##########################################################################################
#= REGISTER SOME CURVES =#

register!("LL Restart", restart_LL_strategy, (color = 1,))
register!("H0 Restart", restart_H0_strategy, (color = 2,))
register!("LL Descent", descent_LL_strategy, (color = 3,))
register!("H0 Descent", descent_H0_strategy, (color = 4,))
register!("Warm Up", warmup_strategy, (color = 5,))

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
prefix = "resampling"

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