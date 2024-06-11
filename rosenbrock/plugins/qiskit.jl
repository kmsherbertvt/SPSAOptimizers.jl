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
            spsa.minimize(fn, x0)

            return nfevs, xs, fs, Δxs, accepteds, spsa._smoothed_hessian

        def calibrate(fn, x0, **kwargs):
            return qopt.SPSA.calibrate(fn, x0, **kwargs)

        """
    end

    function trajectory(fn, x0; order=1, seed=0, options...)
        nfevs, xs, fs, Δxs, accepteds, H =
                py"trajectory"(fn, x0; order=order, seed=seed, options...)
        return (
            nfev = [0; nfevs],
            x = [x0[1]; [xi[1] for xi in xs]],
            y = [x0[2]; [xi[2] for xi in xs]],
            f = [fn(x0); fs],
            g = [0.0; Δxs],
            H = H,
            accepteds = accepteds,
        )
    end

    calibrate(args...; kwargs...) = py"calibrate"(args...; kwargs...)

end