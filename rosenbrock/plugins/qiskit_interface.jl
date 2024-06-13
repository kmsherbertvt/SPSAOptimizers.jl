"""
"""
module QiskitInterfacePlugin
    const Float = Float64

    ######################################################################################
    #= ENSURE THAT JULIA'S PYTHON ENVIRONMENT IS SET UP =#
    import Conda, Pkg

    # FETCH SUMMARY OF JULIA'S PYTHON ENVIRONMENT
    global condalist = begin
        orig_stdout = stdout;
        (rd, wr) = redirect_stdout();
        Conda.list();
        redirect_stdout(orig_stdout);
        close(wr);
        read(rd, String);
    end

    # INSTALL QISKIT CODE IF IT ISN'T ALREADY
    if !occursin("qiskit-algorithms", condalist)
        println("""
        ================================================
        Installing `qiskit-algorithms` python package...
        ================================================
        """)
        Conda.pip_interop(true)
        Conda.pip("install", "qiskit-algorithms")

        println("""
        ================================================
        Building PyCall environment...
        ================================================
        """)
        import Pkg; Pkg.build("PyCall")
    end

    ######################################################################################
    #= INTERFACE INTO QISKIT CODE DIRECTLY =#

    using PyCall

    function __init__()
        py"""
        import qiskit_algorithms
        import qiskit_algorithms.optimizers as qopt

        def tracer():
            nfevs = []; xs = []; fs = []; Δxs = []; accepteds = []
            def callback(nfev, x, f, Δx, accepted):
                nfevs.append(nfev)
                xs.append(x)
                fs.append(f)
                Δxs.append(Δx)
                accepteds.append(accepted)
            return callback, (nfevs, xs, fs, Δxs, accepteds)

        def trajectory(fn, x0, seed, a0, alpha, A, c0, gamma, **kwargs):
            callback, traces = tracer()

            qiskit_algorithms.utils.algorithm_globals.random_seed = seed

            get_eta = lambda: qopt.spsa.powerseries(eta=a0, power=alpha, offset=A)
            get_eps = lambda: qopt.spsa.powerseries(eta=c0, power=gamma)

            spsa = qopt.SPSA(
                learning_rate = get_eta,
                perturbation = get_eps,
                **kwargs,
                callback=callback,
            )

            result = spsa.minimize(fn, x0)

            return traces

        """
    end


    ######################################################################################
    #= INTERFACE INTO THE QISKIT INTERFACE OF `SPSAOptimizers` =#

    import SPSAOptimizers.QiskitInterface as QI

    function tracer()
        nfevs = Int[]
        xs = Vector{Float}[]
        fs = Float[]
        Δxs = Float[]
        accepteds = Bool[]
        function callback(nfev, x, f, Δx, accepted)
            push!(nfevs, nfev)
            push!(xs, x)
            push!(fs, f)
            push!(Δxs, Δx)
            push!(accepteds, accepted)
        end
        return callback, (nfevs, xs, fs, Δxs, accepteds)
    end

    function julia_trajectory(fn, x0; seed, a0, alpha, A, c0, gamma, kwargs...)
        (callback, traces) = tracer()

        QI.seed!(seed)

        get_eta = QI.powerseries(; eta=a0, power=alpha, offset=A)
        get_eps = QI.powerseries(; eta=c0, power=gamma)

        spsa = QI.SPSA(length(x0);
            learning_rate = get_eta,
            perturbation = get_eps,
            kwargs...,
            callback=callback,
        )

        result = QI.minimize(spsa, fn, x0)

        return traces
    end


    ######################################################################################
    #= WRITE A DRIVER THAT CALLS EITHER INTERFACE =#
    function trajectory(
        mode, fn, x0;
        seed=0, a0=0.1, alpha=0.602, A=10.0, c0=0.1, gamma=0.101,
        options...
    )
        (nfevs, xs, fs, Δxs, accepteds) =
            mode == :python ? py"trajectory"(
                fn, x0;
                seed=seed, a0=a0, alpha=alpha, A=A, c0=c0, gamma=gamma,
                options...
            ) :
            mode == :julia ? julia_trajectory(
                fn, x0;
                seed=seed, a0=a0, alpha=alpha, A=A, c0=c0, gamma=gamma,
                options...
            ) :
            error("Unsupported mode")

        return (
            nfev = [0; nfevs],
            x = transpose(reduce(hcat, [[x0]; xs])),
            f = [fn(x0); fs],
            g = [0.0; Δxs],
            accepteds = accepteds,
        )
    end

end