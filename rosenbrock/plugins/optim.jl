"""
"""
module OptimPlugin
    const Float = Float64
    using Optim

    function tracer(f_counter, nfevs, xs, fs, Δxs)
        function callback(optstate)
            push!(nfevs, f_counter[])
            push!(xs, optstate.metadata["x"])
            push!(fs, optstate.value)
            push!(Δxs, optstate.g_norm)
            return false
        end
    end

    """
    I'm adding an extra "order" parameter
        as a package-free way of specifying default optimziers.
    I.e. no need for calling scripts to import `Optim`
        if they just want to use GradientDescent or Newton or BFGS.


    """
    function trajectory(fn, x0; order=1, optimizer=nothing, kwargs...)
        isnothing(optimizer) && (
            optimizer = order == 1 ? GradientDescent() :
                        order == 2 ? Newton() :
                                    BFGS()
        )


        f_counter = Ref(0)
        counted_fn = x -> (f_counter[] += 1; fn(x))

        nfevs = Int[]; xs = Vector{Float}[]; fs = Float[]; Δxs = Float[]
        callback = tracer(f_counter, nfevs, xs, fs, Δxs)

        options = Optim.Options(kwargs..., extended_trace=true, callback=callback)
        optimize(counted_fn, x0, optimizer, options)

        return (
            nfev = [0; nfevs],
            x = [x0[1]; [xi[1] for xi in xs]],
            y = [x0[2]; [xi[2] for xi in xs]],
            f = [fn(x0); fs],
            g = [0.0; Δxs],
        )
    end

end