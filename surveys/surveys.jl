"""
"""
module CensusTemplates
    BASIC = (
        # SAMPLE ALLOCATION
        order = [1, 2],
        n = [1, 4, 16, 64],
        # GAIN SEQUENCES
        η0 = [0.01, 0.1, 2π/10, 1.0],
        α = [:minimal, :optimal],
        ApK = [:qiskit, :spall],
        h0 = [:spall],
        γ = [:optimal],
        # BERNOULLI DISTRIBUTION
        NpL = [1.0],
        p = [1.0],
        # OTHER OPTIONS
        trust = [1.0],
        tolerance = [:qiskit, Inf],
    )


    BROAD = (
        # SAMPLE ALLOCATION
        order = [1, 2],
        n = [1, 4, 16, 64],
        # GAIN SEQUENCES
        η0 = [0.01, 0.1, 2π/10, 1.0],
        α = [:minimal, :optimal],
        ApK = [:qiskit, :spall, 0.5, 1.0],
            #= The largest stability constants appear useless.
                Spall's choice is most helpful When η is smaller. =#
        h0 = [:qiskit, :spall],
        γ = [:minimal, :optimal],
        # BERNOULLI DISTRIBUTION
        NpL = [0.5, 1.0],
        p = [0.5, 1.0],
        # OTHER OPTIONS
        trust = [1.0, Inf],
            #= Higher trust implies confidence the function is more quadratic.
                So, harder functions call for lower trust. =#
        tolerance = [:qiskit, Inf],
    )
end

"""
"""
module CensusTools
    export INPUT, CONTROL, DERIVED, OUTPUT, ALL_FIELDS
    export header!, record!
    export expand, run!, instance!

    import SPSAOptimizers
    const Float = Float64

    import Random
    include("functions.jl"); import .TestFunctions: noisy

    const INPUT = [:L, :M, :σ, :seed]
    const CONTROL = [
        :order, :n,
        :η0, :α, :ApK,
        :h0, :γ,
        :NpL, :p,
        :trust, :tolerance,
    ]
    const DERIVED = [:K, :A, :a0, :N]
    const OUTPUT = [:fp, :nfev]
    const ALL_FIELDS = [INPUT..., CONTROL..., DERIVED..., OUTPUT...]

    """ __input__(; inputs...) -> input... """
    __input__(; L::Int, M::Int, σ::Float, seed::Int) = (L=L, M=M, σ=σ, seed=seed)

    """ __control__(input...; survey...) -> control... """
    __control__(L::Int, M::Int, σ::Float, seed::Int; survey...) = (
        dict = Dict(pairs(survey));

        survey[:α] == :optimal  && (dict[:α] = 1.0);
        survey[:α] == :minimal  && (dict[:α] = 0.602);
        survey[:ApK] == :qiskit && (dict[:ApK] = 0.0);
        survey[:ApK] == :spall  && (dict[:ApK] = 0.1);

        survey[:h0] == :qiskit  && (dict[:h0] = 0.2);
        survey[:h0] == :spall   && (dict[:h0] = iszero(σ) ? 0.02 : σ);
        survey[:γ] == :optimal  && (dict[:γ] = 1/6);
        survey[:γ] == :minimal  && (dict[:γ] = 0.101);

        survey[:tolerance] == :qiskit && (dict[:tolerance] = 2σ);

        NamedTuple(field => dict[field] for field in CONTROL)
    )

    """ __output__(; record...) -> output... """
    __output__(; record...) = NamedTuple(field => record[field][] for field in OUTPUT)

    """ __derived__(input...; control...) -> derived... """
    const __derived__(L::Int, M::Int, σ::Float, seed::Int; control...) = (
        K = M ÷ (2*control[:order] * control[:n]);
        A = K * control[:ApK];
        a0 = control[:η0] * (A + 1)^control[:α];
        N = round(Int, L*control[:NpL]);
        (K = K, A = A, a0 = a0, N = N)
    )



    """ Write the first row of a CSV. """
    function header!(io; fields=ALL_FIELDS)
        join(io, fields, "\t")
        write(io, "\n")
        flush(io)
        return io
    end

    """ Write a data row of a CSV. """
    function record!(io; fields=ALL_FIELDS, kwargs...)
        join(io, (kwargs[field] for field in fields), "\t")
        write(io, "\n")
        flush(io)
        return io
    end


    """ Turn a NamedTuple of vectors into a vector of scalar NamedTuples. """
    function expand(template)
        fields = collect(keys(template))
        lengths = [length(template[field]) for field in fields]
        indices = ones(Int, length(fields))

        expanded = []
        for _ in 1:prod(lengths)
            # ADD THE NEXT NamedTuple
            push!(expanded, NamedTuple(
                fields[i] => template[fields[i]][indices[i]]
                    for i in eachindex(fields)
            ))

            # ITERATE THE INDICES
            indices[1] += 1
            for i in 1:length(indices)-1
                if indices[i] > lengths[i]
                    indices[i] = 1
                    indices[i+1] += 1
                end
            end
            # NOTE: Once `indices[end] == lengths[end]`, we've hit `prod(lengths)`.
        end

        return expanded
    end

    """ Run every combination in a template and write tab-delimited results. """
    function run!(
        io, template, fn, x0;
        M::Int, σ::Float=0.0, seed::Int=0,
        writeheader=true,
    )
        writeheader && header!(io)                                      # I/O!

        # ADD NOISE TO THE GIVEN FUNCTION
        rng = Random.MersenneTwister(seed)
        noisy_fn = noisy(rng, fn, σ)

        # MAKE A LIST OF ALL THE JOBS WE'RE GOING TO DO
        census = expand(template)
        count = length(census)

        # DO THE JOBS
        input = __input__(L=length(x0), M=M, σ=σ, seed=seed)
        for (i, survey) in enumerate(census)
            control = __control__(input...; survey...)
            derived = __derived__(input...; control...)

            SPSA = control.order == 1 ? SPSAOptimizers.SPSA1 : SPSAOptimizers.SPSA2
            record = SPSAOptimizers.Record(SPSA(length(x0)), length(x0))

            costtag = "o=$(control.order), n=$(control.n), K=$(derived.K)"
            println("Starting instance $i/$count ($costtag):")
            try
                @time instance!(                                        # CPU!
                    record, SPSA, noisy_fn, x0;
                    input..., control..., derived...,
                )
            catch e
                @warn """
                --- ERROR ENCOUNTERED in instance $i/$count ($costtag) ---
                $e
                """
            end

            output = __output__(; record...)
            record!(io; input..., control..., derived..., output...)    # I/O!
        end

        return io
    end

    """ Run an instance of SPSA and return the final result. """
    function instance!(record, SPSA, fn, x0;
        K::Int,                             # NUMBER OF ITERATIONS
        a0::Float, α::Float, A::Float,      # LEARNING RATE
        h0::Float, γ::Float,                # PERTURBATION
        N::Int, p::Float, seed::Int,        # RANDOM STREAM
        n::Int,                             # RESAMPLING
        trust::Float, tolerance::Float,     # TRAJECTORY
        unused...
        #= NOTE: This last splat is a lazy trick to let us call
            instance!(record, SPSA, fn, x0; input..., control..., derived...) =#
    )
        L = length(x0)

        η = SPSAOptimizers.PowerSeries(a0, α, A)
        h = SPSAOptimizers.PowerSeries(h0, γ)
        e = SPSAOptimizers.BernoulliDistribution(L=L, k=N, p=p, seed=seed)
        n_= SPSAOptimizers.IntDictStream(default=n)

        spsa = SPSA(L; η=η, h=h, e=e, n=n_)
        return SPSAOptimizers.optimize!(
            spsa, fn, x0;
            maxiter = K,
            trust = trust,
            tolerance = tolerance,
            record = record,
        )
    end

end













# module DEPRECATED
#     # TODO: For backward compatibility with data we pry won't keep.
#     const TRACED_OUTPUTS = [
#         :nfev,
#         :f0, :f1, :f2, :f3, :minf,
#     ]
#     const RECORDS = [INPUTS..., CONTROLS..., TRACED_OUTPUTS...]

#     function cummin(x::AbstractVector)
#         result = similar(x)
#         min_sofar = typemax(eltype(x))
#         for i in eachindex(x)
#             min_sofar = min(min_sofar, x[i])
#             result[i] = min_sofar
#         end
#         return result
#     end

#     function traced_instance(fn, x0;
#         # EXPERIMENTAL PARAMETERS (aka beyond our control)
#         σ::Float, M::Int,
#         # SIMULATION PARAMETERS (aka how we choose to run SPSA)
#         order::Int, n::Int,
#         η0::Float, α::Float, ApK::Float,
#         h0::Float, γ::Float,
#         NpL::Float,
#         trust::Float, tolerance::Float,
#         seed::Int,
#     )
#         fn_rng = Random.MersenneTwister(seed)
#         noisy_fn = noisy(fn_rng, fn, σ)

#         SPSA = order == 1 ? SPSAOptimizers.SPSA1 : SPSAOptimizers.SPSA2

#         K = M ÷ (2*order * n)
#         L = length(x0)
#         A = ApK * K
#         a0 = η0 * (A + 1)^α

#         N = round(Int, NpL*L)

#         η = SPSAOptimizers.PowerSeries(a0, α, A)
#         h = SPSAOptimizers.PowerSeries(h0, γ)
#         e = SPSAOptimizers.BernoulliDistribution(L=L, k=N, seed=seed)
#         n = SPSAOptimizers.IntDictStream(default=n)

#         spsa = SPSA(L; η=η, h=h, e=e, n=n)
#         trace = SPSAOptimizers.Trace()
#         SPSAOptimizers.optimize!(spsa, fn, x0;
#             maxiter = K,
#             trust = trust,
#             tolerance = tolerance,
#             trace = trace,
#             tracefields = (:fp, :nfev),
#         )

#         f = SPSAOptimizers.trajectory(trace, :fp)
#         nfev = SPSAOptimizers.trajectory(trace, :nfev)

#         minf = cummin(f)

#         return (
#             f0 = minf[begin],
#             f1 = minf[1 * length(minf)÷4],
#             f2 = minf[2 * length(minf)÷4],
#             f3 = minf[3 * length(minf)÷4],
#             minf = last(minf),  # aka `f4`, but `minf` is more semantic.
#             nfev = last(nfev),  # Kinda assuming `nfev` is always linear.
#         )
#     end

#     """
#     Returns a JSON-serializable object, to be immediately written by an IO driver.
#     """
#     function traced_run_all!(io, survey, fn, x0; σ::Float, M::Int)
#         expanded = iterator(survey)
#         ntotal = length(expanded)
#         for (i, controls) in enumerate(expanded)
#             println("Starting instance $i/$ntotal:")
#             @time outputs = instance(fn, x0; σ=σ, M=M, controls...)
#             inputs = (L=length(x0), σ=σ, M=M)
#             record!(io, fields=RECORD, inputs..., controls..., outputs...)
#         end

#         return records
#     end


# end