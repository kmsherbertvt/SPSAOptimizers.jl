module FirstOrderOptimizers
    import ..Optimizers

    """
        FirstOrderOptimizer{F}

    Super-type for first-order optimizers.
    (I'm pretty sure there's just the one, `SPSA1`.)
    Constructors are documented with the concrete type(s?),
        but find details on optimization options,
        record schematics, and `iterate!` keyword arguments here:

    # `Record` Schematic

        Record(::FirstOrderOptimizer{F}, L::Int)

    Constructs a `NamedTuple` with fields
    - x::Vector{F} of length L, the current parameters
    - f::Ref{F}, the current function value
    - g::Vector{F} of length L, the current gradient estimate
    - p::Vector{F} of length L, the current parameter step (based on gradient)
    - xp::Vector{F} of length L, the proposed parameters for the next iteration
    - fp::Ref{F}, the function value at the next iteration
    - nfev::Ref{Int}, the number of function evaluations to date
    - time::Ref{F}, the algorithm time to date
    - bytes::Ref{F}, the memory consumption to date

    # Additional `optimize!` Keyword Arguments
    - trust::F=Inf, the maximum allowable norm of `p`

        If `p` has a larger norm, it gets clipped to `trust`.

    - tolerance::F=Inf, the amount by which `fp` may exceed `f` without rejecting the step

    # `iterate!` Interface

        iterate!(
            optimizer, fn, x;
            # EXTRA OPTIONS
            trust,
            # TRACEABLES
            f, g, p, xp, nfev,
            # WORK ARRAYS
            __xe
        )

    - optimizer, fn, x: the mandatory positional arguments, see `iterate!`
    - trust: see above
    - f, g, p, xp, nfev: when provided, stores meaningful outputs
        (when called through `optimize!`,
            these are provided from the `record` of the last successful iteration).
    - __xe: a work variable matching the dimensions of x,
        used to store the stochastically perturbed parameters for finite difference

    """
    abstract type FirstOrderOptimizer{F} <: Optimizers.OptimizerType{F} end

    function Optimizers.Record(::FirstOrderOptimizer{F}, L::Int) where {F}
        return (
            x = zeros(F, L),
            f = Ref(zero(F)),
            g = zeros(F, L),
            p = zeros(F, L),
            xp = zeros(F, L),
            fp = Ref(zero(F)),
            nfev = Ref(zero(Int)),
            time = Ref(zero(F)),
            bytes = Ref(zero(Int)),
        )
    end

    function Optimizers.optimize!(
        optimizer::FirstOrderOptimizer{F}, fn, x0;
        maxiter = 100,
        callback = nothing,
        # EXTRA OPTIONS
        trust = typemax(F),
        tolerance = typemax(F),
        # OUTPUT
        record = Optimizers.Record(optimizer, length(x0)),
        average_last = nothing,
        trace = nothing,
        tracefields = (:f, :fp, :nfev, :time, :bytes),
    ) where {F}
        # INITIALIZE THE "ACTUAL RESULT" RECORD
        #= Setting `xp` and `fp` are conceptually the important part,
            since they inform the "next step" to try. =#
        record.xp  .= x0
        record.fp[] = fn(record.xp); record.nfev[] += 1
        #= But I'll also set `x` and `f` here, so that they are meaningful
            in the event that no successful steps are ever taken. =#
        record.x  .= record.xp
        record.f[] = record.fp[]

        # INITIALIZE THE TRACE
        #= NOTE: Fields x, f, g, p don't mean much in this first record. =#
        isnothing(trace) || Optimizers.trace!(trace, record, tracefields...)

        # ALLOCATE SPACE FOR THE LAST `average_last` RUNS
        if !isnothing(average_last)
            recents = [Optimizers.Record(optimizer, length(x0)) for _ in 1:average_last]
            cursor = 0      # Last index set.
            filled = false  # Whether or not all records have been set once.
        end

        # PREPARE A "THIS INSTANT" RECORD
        iterate = deepcopy(record)
        iterate.x .= x0

        # INITIALIZE WORK VARIABLES
        __xe = similar(x0)

        # RUN THE OPTIMIZATION
        for iter in 1:maxiter
            # DELEGATE MOST OF THE WORK TO `iterate!`
            timing = @timed Optimizers.iterate!(
                optimizer, fn, iterate.x;
                # EXTRA OPTIONS
                trust = trust,
                # TRACEABLES
                f = iterate.f,
                g = iterate.g,
                p = iterate.p,
                xp = iterate.xp,
                nfev = iterate.nfev,
                # WORK VARS
                __xe = __xe,
            )

            # SET THOSE FIELDS NOT MODIFIED BY `iterate!`
            iterate.time[] = timing.time
            iterate.bytes[] = timing.bytes
            iterate.fp[] = fn(iterate.xp); iterate.nfev[] += 1
            #= NOTE: Technically, this extra function evaluation
                is redundant to the trajectory when tolerance=Inf,
                and qiskit avoids it unless a callback is provided.
            But it is recommended by Spall,
                and I can't imagine using qiskit WITHOUT the callback,
                so I'm just making it standard. =#

            # RECORD PROGRESS (even if we wind up rejecting this step!)
            isnothing(trace) || Optimizers.trace!(trace, iterate, tracefields...)

            # CALL CALLBACK (terminate if it returns `true`)
            isnothing(callback) || !callback(optimizer, iterate) || continue

            # DECIDE WHETHER TO REJECT THE STEP
            iterate.fp[] - record.fp[] < tolerance || continue
            #= NOTE: At this point, `iterate.x` remains unchanged,
                    but the state of the optimizer has changed.
                Thus, the next `iterate!` will change `iterate.xp`,
                    which may result in a lower loss function. =#

            # UPDATE THE "ACTUAL RESULT" RECORD
            Optimizers.copyrecord!(record, iterate)

            # UPDATE RECENTS
            if !isnothing(average_last)
                cursor += 1
                if cursor > average_last
                    filled = true
                    cursor = 1
                end
                Optimizers.copyrecord!(recents[cursor], record)
            end

            # ADVANCE TO THE NEXT POINT
            iterate.x .= iterate.xp
        end

        # AVERAGE ALL `recents` INTO `record`
        if !isnothing(average_last) && cursor > 0
            n = filled ? average_last : cursor
            Optimizers.averagerecords!(record, recents[1:n])
        end

        return record
    end
end

module SPSA1s
    import ..Float
    import ..Serialization
    import ..Optimizers

    import ..Streams
    import LinearAlgebra

    import ..FirstOrderOptimizers: FirstOrderOptimizer
    import ..Streams: StreamType
    import FiniteDifferences: FiniteDifferenceMethod

    import FiniteDifferences: central_fdm
    import ..BernoulliDistribution
    import ..PowerSeries
    import ..IntDictStream

    """
        SPSA1{P}(η, h, e, n)

    The first-order SPSA optimizer.

    # Type Parameters
    - P::Int, the order of finite difference, typically 2

    # Parameters
    - η - the float stream for step-length at each iteration
    - h - the float stream for finite-difference perturbation at each iteration
    - e - the vector stream for each finite difference perturbation direction
        This also determines the dimensions in the parameter vector.
    - n - the int stream for number of times to sample the gradient at each iteration



        SPSA1(L; P, η, h, e, n)

    With this constructor, you need only provide the number of dimensions `L`,
        and all the above parameters will be filled in by sensible defaults
        (but you can override any of them with the kwarg).

    Additionally, you may use the `η` and `h` kwargs to specify tuples
        defining a power series.

    # Defaults
    - P: 2, of course - central finite difference
    - η: a power series with a0 = 0.2, alpha=0.602, and A = 0
    - h: a power series with c0 = 0.2 and gamma=0.101
    - e: a fair coin-toss between +/-1 for each dimension
    - n: one sample for every iteration

    # Tuple Interface
    If you don't feel like manually constructing the PowerStream object,
        you can just pass a tuple directly here.

    - η: `(a0, alpha, A)` all floats (even though A is semantically an integer)
    - h: `(c0, gamma)` all floats (note A is always 0 for the h stream)

    """
    struct SPSA1{P} <: FirstOrderOptimizer{Float}
        η::StreamType{Float}
        h::StreamType{Float}
        e::StreamType{Vector{Float}}
        n::StreamType{Int}

        __cfd_0::FiniteDifferenceMethod{P,0}
        __cfd_1::FiniteDifferenceMethod{P,1}

        SPSA1{P}(
            η::Streams.StreamType{Float},
            h::Streams.StreamType{Float},
            e::Streams.StreamType{Vector{Float}},
            n::Streams.StreamType{Int},
        ) where {P} = new{P}(η, h, e, n, central_fdm(P,0), central_fdm(P,1))
    end

    function SPSA1(
        L::Int;
        P = 2,
        η = nothing,
        h = nothing,
        e = nothing,
        n = nothing,
    )
        isnothing(η) && (η = (0.2, 0.602, 0.0))
        isnothing(h) && (h = (0.2, 0.101))
        isnothing(e) && (e = BernoulliDistribution(L=L))
        isnothing(n) && (n = IntDictStream(default=1))

        η isa Tuple && (η = PowerSeries(a0=η[1], γ=η[2], A=η[3]))
        h isa Tuple && (h = PowerSeries(a0=h[1], γ=h[2]))

        return SPSA1{P}(η, h, e, n)
    end

    Serialization.__register__(SPSA1)

    Serialization.__data__(spsa::SPSA1{P}) where {P} = (
        P = P,
        η = Serialization.serialize(spsa.η),
        h = Serialization.serialize(spsa.h),
        e = Serialization.serialize(spsa.e),
        n = Serialization.serialize(spsa.n),
    )

    Serialization.init(::Type{SPSA1}, data) = Serialization.reset!(SPSA1{data.P}(
        Serialization.deserialize(data.η),
        Serialization.deserialize(data.h),
        Serialization.deserialize(data.e),
        Serialization.deserialize(data.n),
    ))

    Serialization.load!(spsa::SPSA1, data) = (
        Serialization.load!(spsa.η, data.η.data);
        Serialization.load!(spsa.h, data.h.data);
        Serialization.load!(spsa.e, data.e.data);
        Serialization.load!(spsa.n, data.n.data);
        spsa
    )

    Serialization.reset!(spsa::SPSA1) = (
        Serialization.reset!(spsa.η);
        Serialization.reset!(spsa.h);
        Serialization.reset!(spsa.e);
        Serialization.reset!(spsa.n);
        spsa
    )

    function Optimizers.iterate!(
        spsa::SPSA1{P}, fn, x;
        # EXTRA OPTIONS
        trust = typemax(Float),
        # TRACEABLES
        f::Ref{Float}=Ref(zero(Float)),
        g::Vector{Float}=Vector{Float}(undef, length(x)),
        p::Vector{Float}=Vector{Float}(undef, length(x)),
        xp::Vector{Float}=Vector{Float}(undef, length(x)),
        nfev::Ref{Int}=Ref(0),
        # WORK ARRAYS
        __xe::Vector{Float}=Vector{Float}(undef, length(x)),
    ) where {P}
        f[] = 0
        g  .= 0

        # Stochastically estimate the function and its gradient at this point.
        n = Streams.next!(spsa.n)
        h = Streams.next!(spsa.h)
        for _ in 1:n
            e = Streams.next!(spsa.e)

            for i in 1:P
                # PERTURB THE PARAMETERS
                __xe .= x .+ h .* spsa.__cfd_0.grid[i] .* e

                # EVALUATE THE PERTURBED COST-FUNCTION
                fe = fn(__xe); nfev[] += 1

                # ESTIMATE THE COST-FUNCTION AND GRADIENT
                f[] += spsa.__cfd_0.coefs[i] * fe
                g  .+= spsa.__cfd_1.coefs[i] .* fe .* e ./ h
            end
        end
        f[] /= n
        g  ./= n

        # First-order methods: descent is along gradient.
        p .= g

        # Enforce trust region.
        if trust < typemax(Float)
            norm = LinearAlgebra.norm(p)
            if norm > trust
                p .*= (trust / norm)
            end
        end

        # Update parameters.
        η = Streams.next!(spsa.η)
        xp .= x .- η .* p

        return xp
    end
end