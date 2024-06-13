#= Question:

Given a fixed budget, is there any advantage to scaling n up or down as you go?

Fix L=20 and fn=RB, and M=(something): each "registration" takes (M,O), selects K=√(M/O), and then generates three curves: n constant throughout, n ramping up linearly, and n ramping down linearly. That should give each curve the same total cost.

A budget of M=20000 nicely sets K[O=1] to 140, and K[O=2] to 100, quite reasonable given that n much larger did absolutely nothing. :) And the uniform n run lands just about where there were diminishing returns in the budget-agnostic runs, so that's great.

Other than comparing O, we can ALSO compare DIFFERENT INITIAL POINTS.
Best to try one in the "gorge" but somewhat closer,
    and one very near the minimum,
    and one rather far off.
Working out what those points are may not be easy.

=#

import SPSAOptimizers

learning_rate(η0, α, A) = SPSAOptimizers.PowerSeries(η0 * (A+1)^α, α, A)

function make_n(mode, K)
    if mode == 0        # FLAT
        return SPSAOptimizers.IntDictStream(default=K÷2)
    elseif mode == 1    # ASCENDING
        dict = Dict(k => k for k in 1:K)
    elseif mode == -1   # DESCENDING
        dict = Dict(k => 1+K-k for k in 1:K)
    else
        error("Invalid mode.")
    end
    return SPSAOptimizers.IntDictStream(dict=dict, default=1)
end

function do_SPSA(fn, x0, O, M, mode; trace=Trace())
    if O == 1
        K = round(Int, √(M/O))
        L = length(x0)
        η = learning_rate(0.1, 1.0, K/10.0)
        h = SPSAOptimizers.PowerSeries(0.1, 1/6)
        e = SPSAOptimizers.BernoulliDistribution(L=L, p=1.0)
        optimizer = SPSAOptimizers.SPSA1(L; η=η, h=h, e=e, n=make_n(mode, K))
        return SPSAOptimizers.optimize!(
            optimizer, fn, x0;
            maxiter = K,
            trust = 1.0,
            tolerance = 0.0,
            trace = trace,
            tracefields = (:nfev, :fp, :xp, :g),
        )
    elseif O == 2
        K = round(Int, √(M/O))
        L = length(x0)
        η = learning_rate(1.0, 1.0, K/10.0)
        h = SPSAOptimizers.PowerSeries(0.1, 1/6)
        e = SPSAOptimizers.BernoulliDistribution(L=L, p=1.0)
        optimizer = SPSAOptimizers.SPSA2(L; η=η, h=h, e=e, n=make_n(mode, K))
        return SPSAOptimizers.optimize!(
            optimizer, fn, x0;
            maxiter = K,
            # trust = 1.0,
            tolerance = 0.0,
            trace = trace,
            tracefields = (:nfev, :fp, :xp, :g),
        )
    else
        error("Invalid O")
    end
end

##########################################################################################

include("plugins/spsa.jl"); import .SPSAPlugin: data_from_trace
function data(fn, x0, O, M, mode)
    trace = Trace()
    do_SPSA(fn, x0, O, M, mode; trace=trace)
    return data_from_trace(trace)
end

include("functions/rosenbrock.jl");
RB = (
    fn = Rosenbrock.lossfunction(),
    gd = Rosenbrock.gradient(),
    hs = Rosenbrock.hessian(),
)

L = 20
xfs = [0.0, 0.5, 0.9, 1.01]

CURVES = Dict()
function register!(xf, O, plot_options)
    x0 = xf .* ones(L)
    label = "xf=$xf O=$O"
    CURVES[label] = (
        label = label,
        xf = xf,
        O = O,
        plot_options = plot_options,
        decline = data(RB.fn, x0, O, 20000, -1),
        uniform = data(RB.fn, x0, O, 20000, 0),
        ramp_up = data(RB.fn, x0, O, 20000, 1),
    )
end

##########################################################################################


for xf in xfs
for O in 1:2
    register!(xf, O, (color=O,))
end; end

##########################################################################################

import Plots
include("plotting/convergence.jl"); import .ConvergencePlots
include("plotting/cost.jl"); import .CostPlots
include("plotting/trajectory.jl"); import .TrajectoryPlots

DEFAULT = (
    markerstrokewidth = 0,
    seriesalpha = 0.5,
    linewidth = 2,
)
DECLINE = (
    markershape = :circle,
    linestyle = :dot,
)
UNIFORM = (
    markershape = :star,
    linestyle = :solid,
)
RAMP_UP = (
    markershape = :x,
    linestyle = :dash,
)

prefix = "rosenbrock/fig/budget.resampling"

plotters = (
    cvg = ConvergencePlots,
    cst = CostPlots,
    trj = TrajectoryPlots,
)
plots = Dict(xf => (
    cvg = ConvergencePlots.init(; log=xf>0.6, nfev=false),
    cst = CostPlots.init(; log=xf>0.6),
    trj = TrajectoryPlots.init(RB.fn; ),
) for xf in xfs)

for xf in xfs
for (name, plt) in pairs(plots[xf])
    plotter = plotters[name]

    for label in sort(collect(keys(CURVES)))
        curve = CURVES[label]
        xf == curve.xf || continue

        plotter.add!(plt, curve.decline;
            DEFAULT..., DECLINE...,
            label=false,
            curve.plot_options...,
        )
        plotter.add!(plt, curve.uniform;
            DEFAULT..., UNIFORM...,
            label=curve.label,
            curve.plot_options...,
        )
        plotter.add!(plt, curve.ramp_up;
            DEFAULT..., RAMP_UP...,
            label=false,
            curve.plot_options...,
        )
    end

    Plots.savefig(plt, "$prefix.$xf.$name.pdf")
end; end
