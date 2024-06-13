"""

Using qiskit, one might do:

    ```

    import qiskit_algorithms
    import qiskit_algorithms.optimizers as qopt

    qiskit_algorithms.utils.algorithm_globals.random_seed = seed

    get_eta = lambda: qopt.spsa.powerseries(eta=a0, power=alpha, offset=A)
    get_eps = lambda: qopt.spsa.powerseries(eta=c0, power=gamma)

    spsa = qopt.SPSA(
        learning_rate = get_eta,
        perturbation = get_eps,
        **kwargs
    )

    result = spsa.minimize(fn, x0)

    ```

With this module, one could do this:

    ```
    import SPSAOptimizers.QiskitInterface as QI

    QI.seed!(seed)

    get_eta = QI.powerseries(; eta=a0, power=alpha, offset=A)
    get_eps = QI.powerseries(; eta=c0, power=gamma)

    spsa = QI.SPSA(length(x0);
        learning_rate = get_eta,
        perturbation = get_eps,
        kwargs...
    )

    result = QI.minimize(spsa, fn, x0)

    ```

Note that `QI.SPSA` needs a positional argument, not needed in Python.
To make up for it, while the `result` object matches qiskit's interface exactly,
    the `spsa` object contains attributes `record` and `trace`,
    which, after the call to `QI.minimize`,
    have some extra information unavailable to qiskit
    (without writing your own callback function).

To avoid expensive and slightly arbitrary calibration steps,
    this module treats `learning_rate` and `perturbation` as mandatory.
Similarly, when the kwarg `blocking` is set to true,
    the kwarg `allowed_increase` also becomes mandatory.
While the qiskit implementation will perform a calibration
    to select problem-informed defaults,
    the calibration is not necessarily ... good.
So you really should perform that calibration ahead of time
    and then manually specify your parameters.

Interesting note: qiskit's `initial_hessian` kwarg seems to be quite useless.
That is, it sets the initial hessian,
    but the initial hessian is, if I have read the code correctly,
    always multiplied by zero before it is ever used.
So.
Don't use it.
If you really want to, I've added an extra kwarg `initial_hessian_weight`
    to the `QI.SPSA` interface, which can be a non-negative integer.
For the record, the `SPSAOptimizers` package does permit an initial hessian,
    but it would requires an extra hyperparameter for an "initial weight",
    so let's just not use it with this `QiskitInterface`.

"""
module QiskitInterface
    import ..Float
    import ..SPSAOptimizers

    import Random
    import LinearAlgebra: I, norm

    const Optional{T} = Union{Nothing,T}

    const QI_RNG = Random.Xoshiro()
    function seed!(seed)
        Random.seed!(QI_RNG, seed)
    end

    function powerseries(; eta = 0.01, power=2.0, offset=0.0)
        return SPSAOptimizers.PowerSeries(eta, power, offset)
    end

    function constant(; eta=0.01)
        return SPSAOptimizers.ConstantSeries(eta)
    end

    function SPSA(L::Int;
        learning_rate,                              # REQUIRED
        perturbation,                               # REQUIRED
        maxiter::Int = 100,
        blocking::Bool = false,
        allowed_increase::Optional{Float} = nothing,# REQUIRED if blocking=true
        trust_region::Bool = false,
        last_avg::Int = 1,
        resamplings::Union{Int,Dict{Int,Int}} = 1,
        perturbation_dims::Optional{Int} = nothing,
        second_order::Bool = false,
        regularization::Optional{Float} = nothing,
        hessian_delay::Int = 0,
        lse_solver = nothing,
        initial_hessian = nothing,                  # BUGGED in qiskit
        callback = nothing,
        termination_checker = nothing,
        initial_hessian_weight::Int = 0,            # RESOLVES `initial_hessian`
    )
        # INPUT VALIDATION
        blocking && isnothing(allowed_increase) && error("""
            Default calibration for `allowed_increase` not supported when `blocking=true`.
        """)

        isnothing(initial_hessian) || error("""
            The `initial_hessian` doesn't work in qiskit, so it's not supported here.
        """)

        # SOME OF THESE `nothing` DEFAULTS REALLY COULD JUST HAVE VALUES...
        isnothing(perturbation_dims) && (perturbation_dims = L)
        isnothing(regularization) && (regularization = 0.01)
        isnothing(lse_solver) && (lse_solver = \)
        isnothing(initial_hessian) && (initial_hessian = Matrix{Float}(I, L, L))

        # PREPARE THE RANDOM STREAM
        random_stream = SPSAOptimizers.BernoulliDistribution(
            L, perturbation_dims, 1.0,
            0,  # Using a "global" RNG, this seed has no effect.
            zeros(L), QI_RNG,
        )
        #= NOTE: If you serialize the optimizer,
            it's going to use the dummy seed, rather than the global RNG,
            when you try to load it, so don't expect trajectories to match. =#

        # PREPARE THE RESAMPLING DICT
        sample_stream = resamplings isa Int ?
            SPSAOptimizers.IntDictStream(default=resamplings) :
            SPSAOptimizers.IntDictStream(dict=resamplings, default=1)

        # PREPARE THE HESSIAN OBJECT (not actually used if `second_order=false`)
        hessian = SPSAOptimizers.TrajectoryHessian(
            regularization,
            initial_hessian,
            Ref(initial_hessian_weight),
        )

        # PREPARE SOME SCALAR PARAMETERS
        trust = trust_region ? 1.0 : typemax(Float)
        tolerance = blocking ? allowed_increase : Inf

        # PREPARE THE CALLBACK
        use_callback = !(isnothing(callback) && isnothing(termination_checker))
        function doubleduty(optimizer, iterate)
            # EXTRACT THE PARAMETERS THAT THE PYTHON CALLBACKS EXPECT
            nfev = iterate.nfev[]
            x = iterate.xp
            f = iterate.fp[]
            Δx = norm(iterate.x .- iterate.xp)
            accepted = blocking || iterate.fp[] - iterate.f[] < tolerance

            # CALL THE DATA CALLBACK
            isnothing(callback) || callback(nfev, x, f, Δx, accepted)

            # CALL THE TERMINATION CALLBACK
            terminate = accepted || isnothing(termination_checker) ? false :
                termination_checker(nfev, x, f, Δx, accepted)
            #= NOTE: Qiskit only calls `termination_checker` on accepted steps.
                I think that's dumb, but I'm trying to do what they do here. =#

            return terminate
        end

        if second_order
            optimizer = SPSAOptimizers.SPSA2(
                hessian,
                learning_rate,
                perturbation,
                random_stream,
                sample_stream,
            )

            record = SPSAOptimizers.Record(optimizer, L)
            trace = SPSAOptimizers.Trace()

            options = (
                maxiter = 100,
                callback = use_callback ? doubleduty : nothing,
                # EXTRA OPTIONS
                trust = trust,
                tolerance = tolerance,
                warmup = hessian_delay,
                solver = lse_solver,
                # OUTPUT
                record = record,
                average_last = last_avg == 1 ? nothing : last_avg,
                trace = trace,
            )
        else
            optimizer = SPSAOptimizers.SPSA1{2}(
                learning_rate,
                perturbation,
                random_stream,
                sample_stream,
            )

            record = SPSAOptimizers.Record(optimizer, L)
            trace = SPSAOptimizers.Trace()

            options = (
                maxiter = 100,
                callback = use_callback ? doubleduty : nothing,
                # EXTRA OPTIONS
                trust = trust_region ? 1.0 : typemax(Float),
                tolerance = blocking ? allowed_increase : typemax(Float),
                # OUTPUT
                record = record,
                average_last = last_avg == 1 ? nothing : last_avg,
                trace = trace,
            )
        end

        return (
            optimizer = optimizer,
            options = options,
            record = record,
            trace = trace,
        )
    end

    function minimize(spsa, fn, x0)
        SPSAOptimizers.optimize!(spsa.optimizer, fn, x0; spsa.options...)
        return (
            x = spsa.record.xp,
            fun = spsa.record.fp[],
            nfev = spsa.record.nfev[],
            nit = length(spsa.trace)-1,
        )
    end
end