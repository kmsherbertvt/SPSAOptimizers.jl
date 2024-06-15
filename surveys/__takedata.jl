include("functions.jl")
    import .TestFunctions

include("surveys.jl")
    import .CensusTemplates
    import .CensusTools

outdir = "surveys/surveys/rosenbrock/zeropoint"
outfile = "$outdir/L20_M20000_σ0.0.csv"

open(outfile, "w") do io
    CensusTools.run!(io,
        CensusTemplates.BASIC,
        TestFunctions.Rosenbrock.lossfunction(),
        zeros(20);
        M=20000,
    )
end



# import JSON

# include("functions.jl"); using .TestFunctions
# include("surveys.jl"); using .SurveyTools

# """
# """
# SURVEY = (
#     # SAMPLE ALLOCATION
#     order = [1, 2],
#         #= NOTE: These are the only two choices.
#             If your function is well approximated by a quadratic function,
#                 second-order is almost certainly the better choice.
#             But ya don't necessarily know that goin' in. =#
#     n = [1, 4, 16, 64],
#         #= NOTE: Lower = more stochastic.
#             That can be a good thing or a bad thing.
#             From my preliminary tests,
#                 convergence per iteration is almost always better with larger n
#                 (for large enough n, the "Stochastic Approximation" in SPSA
#                     approaches "Not An Approximation"),
#                 but convergence per function evaluation often isn't.
#             The sweet spot is something that seems to depend a lot
#                 on the total sample budget. =#
#     # GAIN SEQUENCES
#     η0 = [0.01, 0.1, 2π/10, 1.0],
#         #= NOTE: This one is probably the most important hyperparameter,
#                     and of course it's the one I least understand.

#             Technically, it determines the initial stepsize,
#                 or more precisely,
#                 the fraction of the proposed step that we actually take.
#             Ultimately, it determines whether or not we can find a path
#                 (smaller is better),
#                 and then whether we actually make any progress down it
#                 (larger is better).

#             For second order,
#                 if your function is well approximated by a quadratic function,
#                 and you have a decent approximation of the Hessian,
#                 then you want it at exactly 1.0,
#                 since the proposed step itself is ostensibly "perfect".
#             If these conditions are not satisfied, all bets are off...

#             For first order, Spall recommends a convoluted expensive calibration
#                 to ensure the first step is a desirable distance
#                 from the initial point.
#             Of course, he offers no insight on that desirable distance. :)
#             Qiskit *mostly* implements Spall's calibration
#                 (they forgot to account for the stability constant, I think),
#                 and they use a default "desirable distance" of 2π/10.
#             Note that the qiskit code is oriented around gate-model VQE,
#                 such that the full range of any given parameter is ±π.
#             Thus, if we assume that qiskit's heuristic is useful,
#                 I conjecture that a "desirable distance" for ctrl-VQE is ±ΩMAX.
#             Incidentally, if qiskit's calibration "fails"
#                 (I forget what that even means),
#                 they just set η0 itself to that "desirable distance".
#             I doubt that's a particularly meaningful choice,
#                 but it's somethin' :)
#         =#
#     α = [0.602, 1.0],
#         #= NOTE: Lower = faster, at least at first
#                 (which may be all you get in limited function evals).
#             But 0.602 is as low as you can go with a convergence guarantee.
#             Using 1.0 is theoretically optimal, eventually. =#
#     ApK = [0.0, 0.1, 0.5, 1.0],
#         #= NOTE: Spall advises 0.1 empirically.
#                 I think the idea is to exploit the asymptotic properties of α,
#                 but to stretch out the period of time at which η is not tiny,
#                 so I conjecture that ApK≠0 is supposed to make α=1.0 better.
#             Qiskit uses 0.0 by default. =#
#         # TODO: The initial survey did a whole bunch. Just note I've already trimmed it down to just the two that seem relevant, but we should study the generated results to assess whether there is in fact some value to the others!
#     h0 = [0.02, 0.2],
#         #= NOTE: Spall advises h0=σ, or else a "small positive number".
#             Qiskit uses 0.2 by default.
#             Odd, that. They estimate σ for `tolerance`,
#                 so they ought to use it here too... :/ =#
#         # TODO: Maybe just go ahead and fix this to σ, or 0.01 if σ=0.
#     γ = [0.101, 1/6],
#         #= NOTE: Like α, 0.101 is as low as you can go,
#                 while 1/6 is theoretically optimal.
#             But I don't know if this one matters as much.
#             I'm sure it does with noisy functions. =#
#         # TODO: Please check if this matters at all with a noisy run, and then if not get rid of it.
#     # BERNOULLI DISTRIBUTION
#     NpL = [0.5, 1.0],
#         #= NOTE: Reducing NpL in principle lets SPSA explore more directions,
#                     which you'd think would help when it gets stuck,
#                     but I've never seen this help. =#
#         # TODO: Probably can fix this to 1.0.
#     # OTHER OPTIONS
#     trust = [1.0, Inf],
#         #= NOTE: Qiskit only allows these two choices. =#
#     tolerance = [0.0, Inf],
#         #= NOTE: Qiskit calibrates to 2σ, which seems reasonable. =#
#         # TODO: Maybe just go ahead and fix this to 2σ. But no, I think you're not supposed to run blocking with 1SPSA?
#     # RANDOM NUMBER GENERATION
#     seed = [0],
#         #= NOTE: This defines both function noise and direction choice,
#                     but via independent random streams. =#
# )



# fn = TestFunctions.Rosenbrock.lossfunction()
# L = 20
# x0 = zeros(L)
# outdir = "surveys/surveys/rosenbrock/zeropoint"

# σ = 0.0
#     #= NOTE: This is related to the shot count and other incoherent noise. =#
# M = 20000       # TOTAL NUMBER OF FUNCTION EVALUATIONS
#     #= NOTE: This is less than the total number of CIRCUIT evaluations! =#
# outfile = "$outdir/L$(L)_M$(M)_σ$(σ).json"
# isfile(outfile) && error("File already exists!  $outfile")

# records = []
# records = SurveyTools.run_all!(records, SURVEY, fn, x0; σ=σ, M=M)

# open(outfile, "w") do io
#     JSON.print(io, records, 4)
# end