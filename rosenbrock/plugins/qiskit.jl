"""
"""
module QiskitPlugin
    #=

    This is a bit convoluted and probably a waste of time but I just can't help it...

    This module needs a conda environment with `qiskit-algorithms`.
    We can make that happen manually, if it's not here,
        but we don't want to do the package shuffling if it isn't necessary.
    So, the first step is to check if `qiskit-algorithms` appears in the conda environment.

    =#
    import Conda, Pkg

    global condalist = begin
        orig_stdout = stdout;
        (rd, wr) = redirect_stdout();
        Conda.list();
        redirect_stdout(orig_stdout);
        close(wr);
        read(rd, String);
    end

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

    using PyCall

    function __init__()
        py"""
        import numpy as np
        import qiskit_algorithms as qa
        import qiskit_algorithms.optimizers as qopt

        def tracer(nfevs, xs, fs, Δxs, accepteds):
            def callback(nfev, x, f, Δx, accepted):
                nfevs.append(nfev)
                xs.append(x)
                fs.append(f)
                Δxs.append(Δx)
                accepteds.append(accepted)
            return callback

        def trajectory(fn, x0, order=1, seed=0, **spsa_kwargs):
            if order == 2: spsa_kwargs["second_order"] = True

            nfevs = []; xs = []; fs = []; Δxs = []; accepteds = []
            callback = tracer(nfevs, xs, fs, Δxs, accepteds)

            qa.utils.algorithm_globals.random_seed = seed
            spsa = qopt.SPSA(
                **spsa_kwargs,
                callback=callback,
            )
            result = spsa.minimize(fn, x0)

            return nfevs, xs, fs, Δxs, accepteds, spsa, result

        def calibrate(fn, x0, **kwargs):
            return qopt.SPSA.calibrate(fn, x0, **kwargs)
        def powerseries(eta=0.01, power=2, offset=0):
            return qopt.spsa.powerseries(eta=eta, power=power, offset=offset)
        def wrapped_powerseries(eta=0.01, power=2, offset=0):
            def wrapper():
                return powerseries(eta=eta, power=power, offset=offset)
            return wrapper

        """
    end

    function trajectory(fn, x0; order=1, seed=0, options...)
        nfevs, xs, fs, Δxs, accepteds, optimizer, result =
                py"trajectory"(fn, x0; order=order, seed=seed, options...)
        return (
            nfev = [0; nfevs],
            x = transpose(reduce(hcat, [[x0]; xs])),
            f = [fn(x0); fs],
            g = [0.0; Δxs],
            accepteds = accepteds,
            optimizer = optimizer,
            result = result,
        )
    end

    calibrate(args...; kwargs...) = py"calibrate"(args...; kwargs...)
    powerseries(args...; kwargs...) = py"powerseries"(args...; kwargs...)
    wrapped_powerseries(args...; kwargs...) = py"wrapped_powerseries"(args...; kwargs...)

end