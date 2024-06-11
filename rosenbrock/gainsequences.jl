#= Convergence guarantees are sensitive to the choice of η and h at each iteration.
    Evidently they need to taper off in some way.
    The choices η[k] and h[k] are called "gain sequences".

    (By the way, Spall always uses the letter a for my η, and c for my h.)

Spall qualitatively discusses recommendations for gain sequences here:

    https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

They are summarized here:

Let η[k] = η0 / (A + k + 1)^α
    h[k] = h0 / (k + 1)^γ

- Asymptotically optimal: α=1, γ=1/6
- Lowest allowable for theoretical convergence: α=0.602, γ=0.101 (qiskit's defaults)
- Empirically, the smaller the better,
    but maybe for large problems one should transition to the larger.
- Set h0 to the standard deviation of the noise in f,
    or a "small positive number" if there is no noise.
  There's also a note in another paper recommending "an order of magnitude larger"
    when using second-order SPSA.
- Set A to 10% of the maximum number of iterations.
- Set a0 such that a[0] * |g(x0)| gives a desirable "change in magnitude" of x.
    This is what qiskit's calibration does,
        though they actually omit A from that calculation,
        which I'm fairly sure is a bug.



This script is just going to plot a bunch of these sequences,
    to build intuition for what they mean.

=#

import SPSAOptimizers: Streams, PowerSeries
Float = Float64

K = 100     # MAXIMUM NUMBER OF ITERATIONS

σf = 0.1    # STANDARD DEVIATION OF NOISE IN f
Δx = 0.2    # TARGET MAGNITUDE OF FIRST STEP
g0 = 2.0    # ESTIMATE OF |g| AT FIRST STEP

unstable_learning_rate(α) = (A = 0.0; PowerSeries(Δx / g0 * (A+1)^α, α, A))
stable_learning_rate(α) = (A = K/10; PowerSeries(Δx / g0 * (A+1)^α, α, A))
perturbation(γ) = PowerSeries(σf, γ)

##########################################################################################

curves = Dict{String,Any}()
register!(label, series; plot=true, kwargs...) = (
    curves[label] = (series=series, label=label, plot=plot, kwargs=kwargs)
)

##########################################################################################

register!(
    "Asymptotic η (A=0)",
    unstable_learning_rate(1.0),
    ls=:dot, color=1,
)

register!(
    "Asymptotic η (A≠0)",
    stable_learning_rate(1.0),
    ls=:dot, color=2,
)

register!(
    "Asymptotic h",
    perturbation(1/6),
    ls=:dot, color=3,
)




register!(
    "Greedy η (A=0)",
    unstable_learning_rate(0.602),
    ls=:solid, color=1,
)

register!(
    "Greedy η (A≠0)",
    stable_learning_rate(0.602),
    ls=:solid, color=2,
)

register!(
    "Greedy h",
    perturbation(0.101),
    ls=:solid, color=3,
)





##########################################################################################

fig_dir = "rosenbrock/fig"

import Plots

plt = Plots.plot(;
)

for (label, curve) in pairs(curves)
    curve.plot || continue
    a = [Streams.next!(curve.series) for k in 0:K]
    Plots.plot!(plt, a; label=label, lw=3, curve.kwargs...)
end

Plots.savefig(plt, "$fig_dir/gainsequences.pdf")