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
            x = transpose(reduce(hcat, [[x0]; xs])),
            f = [fn(x0); fs],
            g = [0.0; Δxs],
            optimizer = optimizer,
        )
    end

end