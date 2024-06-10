"""
"""
module SPSAPlugin
    const Float = Float64
    using SPSAOptimizers

    import LinearAlgebra

    #= TODO: Eventually much of this will be wrapped up
        in various options to SPSAOptimizers.optimize!. =#

    function trajectory(fn, x0; order=1, optimizer=nothing, niter=100, options...)
        L = length(x0)
        isnothing(optimizer) && (
            optimizer = order == 1 ? SPSAOptimizers.SPSA(L; options...) :
                        # order == 2 ? SPSAOptimizers.QiskitSPSA2(L; options...) :
                                    error("order ≠ 1 or 2 is not (yet) implemented")
        )

        nfevs = Int[]; xs = Vector{Float}[]; fs = Float[]; Δxs = Float[]

        x = deepcopy(x0)

        nfev = Ref(0)
        f = Ref(0.0)
        g = similar(x)
        p = similar(x)
        for _ in 1:niter
            SPSAOptimizers.Optimizers.iterate!(
                optimizer, x, fn;
                nfev=nfev, f=f, g=g, p=p,
            )

            push!(nfevs, nfev[])
            push!(xs, deepcopy(x))
            push!(fs, f[])
            push!(Δxs, LinearAlgebra.norm(p))
        end

        return (
            nfev = [0; nfevs],
            x = [x0[1]; [xi[1] for xi in xs]],
            y = [x0[2]; [xi[2] for xi in xs]],
            f = [fn(x0); fs],
            g = [0.0; Δxs],
        )
    end

end