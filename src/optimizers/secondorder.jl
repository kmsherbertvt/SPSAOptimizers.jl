module SecondOrderOptimizers
    import ..Optimizers

    """
        SecondOrderOptimizer{F}

    Super-type for first-order optimizers.
    (Just `SPSA2` for now but I have visions of how to enhance.
        They probably aren't worth the effort, though.)
    Constructors are documented with the concrete type(s?),
        but find details on optimization options,
        record schematics, and `iterate!` keyword arguments here:

    # `Record` Schematic

        Record(::FirstOrderOptimizer{F}, L::Int)

    Constructs a `NamedTuple` with fields
    - x::Vector{F} of length L, the current parameters
    - f::Ref{F}, the current function value
    - g::Vector{F} of length L, the current gradient estimate
    - H::Matrix{F} of size(L,L), the current Hessian estimate
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

    - warmup::Int=0, after how many iterations to begin refining the step with the Hessian
        (rather than just using gradient directly)

    - solver, a function with header `solver(H, g)`
        returning the "preconditioned" ``H^{-1} g``

    # `iterate!` Interface

        iterate!(
            optimizer, fn, x;
            # EXTRA OPTIONS
            trust, precondition, solver,
            # TRACEABLES
            f, g, H, p, xp, nfev,
            # WORK ARRAYS
            __xe,
            __e1,
        )

    - optimizer, fn, x: the mandatory positional arguments, see `iterate!`
    - trust, solver: see above
    - precondition: whether or not to use the Hessian in this iteration
    - f, g, H, p, xp, nfev: when provided, stores meaningful outputs
        (when called through `optimize!`,
            these are provided from the `record` of the last successful iteration).
    - __xe: a work variable matching the dimensions of x,
        Used to store the stochastically perturbed parameters for finite difference
    - __e1: a work variable matching the dimensions of x
        Used to store one of the two perturbation vectors
            needed for second order finite difference.
        Note that the other one is stored within the stream object,
            which is why this isn't needed at all in first-order.

    """
    abstract type SecondOrderOptimizer{F} <: Optimizers.OptimizerType{F} end

    function Optimizers.Record(::SecondOrderOptimizer{F}, L::Int) where {F}
        return (
            x = zeros(F, L),
            f = Ref(zero(F)),
            g = zeros(F, L),
            H = zeros(F, L, L),
            p = zeros(F, L),
            xp = zeros(F, L),
            fp = Ref(zero(F)),
            nfev = Ref(zero(Int)),
            time = Ref(zero(F)),
            bytes = Ref(zero(Int)),
        )
    end

    function Optimizers.optimize!(
        optimizer::SecondOrderOptimizer{F}, fn, x0;
        maxiter = 100,
        callback = nothing,
        # EXTRA OPTIONS
        trust = typemax(F),
        tolerance = typemax(F),
        warmup = 0,
        solver = \,
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
        #= NOTE: Fields x, f, g, H, p don't mean much in this first record. =#
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
        __e1 = similar(x0)

        # RUN THE OPTIMIZATION
        for iter in 1:maxiter
            # DELEGATE MOST OF THE WORK TO `iterate!`
            timing = @timed Optimizers.iterate!(
                optimizer, fn, iterate.x;
                # EXTRA OPTIONS
                trust = trust,
                precondition = iter > warmup,
                solver = solver,
                # TRACEABLES
                f = iterate.f,
                g = iterate.g,
                H = iterate.H,
                p = iterate.p,
                xp = iterate.xp,
                nfev = iterate.nfev,
                # WORK VARS
                __xe = __xe,
                __e1 = __e1,
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

module SPSA2s
    import ..Float
    import ..Serialization
    import ..Optimizers

    import ..Hessians
    import ..Streams
    import LinearAlgebra

    import ..SecondOrderOptimizers: SecondOrderOptimizer
    import ..Hessians: HessianType
    import ..Streams: StreamType

    import ..TrajectoryHessian
    import ..BernoulliDistribution
    import ..PowerSeries
    import ..IntDictStream


    """
        SPSA2(H, η, h, e, n)

    The second-order SPSA optimizer.

    # Parameters
    - H - the Hessian object, updated at each iteration
    - η - the float stream for step-length at each iteration
    - h - the float stream for finite-difference perturbation at each iteration
    - e - the vector stream for each finite difference perturbation direction
        This also determines the dimensions in the parameter vector.
    - n - the int stream for number of times to sample the gradient at each iteration



        SPSA2(L; H, η, h, e, n)

    With this constructor, you need only provide the number of dimensions `L`,
        and all the above parameters will be filled in by sensible defaults
        (but you can override any of them with the kwarg).

    Additionally, you may use the `η` and `h` kwargs to specify tuples
        defining a power series.

    # Defaults
    - H: a fresh LxL `TrajectoryHessian` with regularization bias 0.01
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
    struct SPSA2 <: SecondOrderOptimizer{Float}
        H::HessianType{Float}
        η::StreamType{Float}
        h::StreamType{Float}
        e::StreamType{Vector{Float}}
        n::StreamType{Int}
    end

    function SPSA2(
        L::Int;
        H = nothing,
        η = nothing,
        h = nothing,
        e = nothing,
        n = nothing,
    )
        isnothing(H) && (H = TrajectoryHessian(L))
        isnothing(η) && (η = (0.2, 0.602))
        isnothing(h) && (h = (0.2, 0.602))
        isnothing(e) && (e = BernoulliDistribution(L=L))
        isnothing(n) && (n = IntDictStream(default=1))

        η isa Tuple && (η = PowerSeries(a0=η[1], γ=η[2]))
        h isa Tuple && (h = PowerSeries(a0=h[1], γ=h[2]))

        return SPSA2(H, η, h, e, n)
    end

    Serialization.__register__(SPSA2)

    Serialization.__data__(spsa::SPSA2) = (
        H = Serialization.serialize(spsa.H),
        η = Serialization.serialize(spsa.η),
        h = Serialization.serialize(spsa.h),
        e = Serialization.serialize(spsa.e),
        n = Serialization.serialize(spsa.n),
    )

    Serialization.init(::Type{SPSA2}, data) = Serialization.reset!(SPSA2(
        Serialization.deserialize(data.H),
        Serialization.deserialize(data.η),
        Serialization.deserialize(data.h),
        Serialization.deserialize(data.e),
        Serialization.deserialize(data.n),
    ))

    Serialization.load!(spsa::SPSA2, data) = (
        Serialization.load!(spsa.H, data.H.data);
        Serialization.load!(spsa.η, data.η.data);
        Serialization.load!(spsa.h, data.h.data);
        Serialization.load!(spsa.e, data.e.data);
        Serialization.load!(spsa.n, data.n.data);
        spsa
    )

    Serialization.reset!(spsa::SPSA2) = (
        Serialization.reset!(spsa.H);
        Serialization.reset!(spsa.η);
        Serialization.reset!(spsa.h);
        Serialization.reset!(spsa.e);
        Serialization.reset!(spsa.n);
        spsa
    )

    function Optimizers.iterate!(
        spsa::SPSA2, fn, x;
        # EXTRA OPTIONS
        trust = typemax(Float),
        precondition = true,
        solver = \,
        # TRACEABLES
        f::Ref{Float}=Ref(zero(Float)),
        g::Vector{Float}=Vector{Float}(undef, length(x)),
        H::Matrix{Float}=Matrix{Float}(undef, length(x), length(x)),
        p::Vector{Float}=Vector{Float}(undef, length(x)),
        xp::Vector{Float}=Vector{Float}(undef, length(x)),
        nfev::Ref{Int}=Ref(0),
        # WORK ARRAYS
        __xe::Vector{Float}=Vector{Float}(undef, length(x)),
        __e1::Vector{Float}=Vector{Float}(undef, length(x)),
    )
        f[] = 0
        g  .= 0
        H  .= 0

        # Stochastically estimate the function, gradient, and Hessian.
        n = Streams.next!(spsa.n)
        h = Streams.next!(spsa.h)
        for _ in 1:n
            __e1 .= Streams.next!(spsa.e)
            e2 = Streams.next!(spsa.e)

            wt_H = zero(Float)

            # +e1, 0⋅e2
            __xe .= x .+ h .* __e1
            fe = fn(__xe); nfev[] += 1
            f[] += fe / 4
                #= NOTE: Divide by four here (rather than two) because qiskit
                    includes second-order measurements in the average.
                    I don't think it should... :/ =#
            g  .+= fe .* __e1 ./ 2h
            wt_H -= fe / 2h^2

            # -e1, 0⋅e2
            __xe .= x .- h .* __e1
            fe = fn(__xe); nfev[] += 1
            f[] += fe / 4
            g  .-= fe .* __e1 ./ 2h
            wt_H += fe / 2h^2

            # +e1, +e2
            __xe .= x .+ h .* (__e1 .+ e2)
            fe = fn(__xe); nfev[] += 1
            f[] += fe / 4
            wt_H += fe / 2h^2

            # -e1, +e2
            __xe .= x .- h .* (__e1 .- e2)
            fe = fn(__xe); nfev[] += 1
            f[] += fe / 4
            wt_H -= fe / 2h^2

            # ADD THE RANK ONE TENSOR `wt_H * outer(e1 e2)`, symmetrized
            LinearAlgebra.mul!(H, __e1, e2', wt_H / 2, one(Float))
            LinearAlgebra.mul!(H, e2, __e1', wt_H / 2, one(Float))
        end
        f[] /= n
        g  ./= n
        H  ./= n

        # Update smoothed Hessian.
        Hessians.update!(spsa.H, H)

        # Combine Hessian and gradient to decide the actual descent.
        p .= precondition ? solver(Hessians.regularized(spsa.H), g) : g
            #= NOTE: I feel like there is no way to avoid allocations here,
                so I'm not even trying. =#

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

        return false
    end
end


#= TODO: MySPSA2: Use an order-P finite difference,
    requiring P^2 function evals per iteration for the Hessian estimate
    (plus P for the function and gradient, as in SPSA1). =#

#= TODO: BFGSSPSA: Use a BFGS-like update to approximate Hk rather than H.
    I believe this obviates any need for extra measurements,
        and it certainly obviates the need for `solver`,
        so presumably this deserves its own file.
    That is, it's not a `SecondOrderOptimizer` at all.

    It may be that BFGS-like updates need a linesearch to perform well,
        and I ain't doin' that, so I dunno if this should be expected to work.
    But *I* expect it ought to. :) =#