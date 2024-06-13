"""
"""
module SPSAPlugin
    const Float = Float64
    using SPSAOptimizers

    import LinearAlgebra

    function trajectory(fn, x0; order=1,
        optimizer = nothing,
        optimizeroptions=NamedTuple(),
        options...
    )
        L = length(x0)
        isnothing(optimizer) && (
            optimizer =
                order == 1 ? SPSAOptimizers.SPSA1(L; optimizeroptions...) :
                order == 2 ? SPSAOptimizers.SPSA2(L; optimizeroptions...) :
                error("order ≠ 1 or 2 is not (yet) implemented")
        )

        record = SPSAOptimizers.Record(optimizer, L)
        trace = SPSAOptimizers.Trace()

        SPSAOptimizers.optimize!(
            optimizer, fn, x0;
            options...,
            record = record,
            trace = trace,
            tracefields = Symbol[],
        )

        return (
            nfev = SPSAOptimizers.trajectory(trace, :nfev),
            x = transpose(SPSAOptimizers.trajectory(trace, :xp)),
            f = SPSAOptimizers.trajectory(trace, :fp),
            g = LinearAlgebra.norm.(SPSAOptimizers.trajectory(trace, :g)),
            optimizer = optimizer,
            record = record,
            trace = trace,
        )

    end

    function data_from_trace(trace)
        return (
            nfev = SPSAOptimizers.trajectory(trace, :nfev),
            x = transpose(SPSAOptimizers.trajectory(trace, :xp)),
            f = SPSAOptimizers.trajectory(trace, :fp),
            g = LinearAlgebra.norm.(SPSAOptimizers.trajectory(trace, :g)),
        )
    end

end