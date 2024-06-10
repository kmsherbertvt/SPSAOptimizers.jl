"""
"""
module FirstOrderOptimizers
    import ..Float
    import ..Serialization
    import ..Optimizers

    import ..Streams

    import Parameters: @with_kw

    import LinearAlgebra

    import FiniteDifferences: FiniteDifferenceMethod, central_fdm
    import ..BernoulliDistribution
    import ..PowerSeries

    """
    """
    @with_kw struct SPSA <: Optimizers.OptimizerType{Float}
        η::Streams.StreamType{Float}
        h::Streams.StreamType{Float}
        e::Streams.StreamType{Vector{Float}}
        p::Int = 2
        n::Int = 1
        trust_region::Float = typemax(Float)

        __cfd_0 = central_fdm(p,0)
        __cfd_1 = central_fdm(p,1)
    end

    function SPSA(
        L::Int;
        η = nothing,
        h = nothing,
        e = nothing,
        p = 2,
        n = 1,
        trust_region = typemax(Float),
    )
        isnothing(η) && (η = (0.2, 0.602))
        isnothing(h) && (h = (0.2, 0.602))
        isnothing(e) && (e = BernoulliDistribution(L=L))

        η isa Tuple && (η = PowerSeries(a0=η[1], γ=η[2]))
        h isa Tuple && (h = PowerSeries(a0=h[1], γ=h[2]))

        return SPSA(η=η, h=h, e=e, p=p, n=n, trust_region=trust_region)
    end

    Serialization.__register__(SPSA)

    Serialization.__data__(spsa::SPSA) = (
        η = Serialization.__data__(spsa.η),
        h = Serialization.__data__(spsa.h),
        e = Serialization.__data__(spsa.e),
        p = spsa.p,
        n = spsa.n,
        trust_region = spsa.trust_region,
    )

    Serialization.init(::Type{SPSA}, data) = Serialization.reset!(SPSA(
        η = Serialization.deserialize(data.η),
        h = Serialization.deserialize(η),
        e = Serialization.deserialize(η),
        p = data.p,
        n = data.η,
        trust_region = data.trust_region,
    ))

    Serialization.load!(spsa::SPSA, data) = (
        Serialization.load!(spsa.η, data.η);
        Serialization.load!(spsa.h, data.h);
        Serialization.load!(spsa.e, data.e);
        spsa
    )

    Serialization.reset!(spsa::SPSA) = (
        Serialization.reset!(spsa.η);
        Serialization.reset!(spsa.h);
        Serialization.reset!(spsa.e);
        spsa
    )

    """
    """
    function Optimizers.iterate!(
        spsa::SPSA, x, fn;
        nfev::Ref{Int}=Ref(0),
        f::Ref{Float}=Ref(zero(Float)),
        g::Vector{Float}=Vector{Float}(undef, length(x)),
        p::Vector{Float}=Vector{Float}(undef, length(x)),
        __xe::Vector{Float}=Vector{Float}(undef, length(x)),
    )
        f[] = 0
        g  .= 0

        # Stochastically estimate the function and its gradient at this point.
        h = Streams.next!(spsa.h)
        for _ in 1:spsa.n
            e = Streams.next!(spsa.e)

            for i in 1:spsa.p
                # PERTURB THE PARAMETERS
                __xe .= x .+ h .* spsa.__cfd_0.grid[i] .* e

                # EVALUATE THE PERTURBED COST-FUNCTION
                fe = fn(__xe); nfev[] += 1

                # ESTIMATE THE COST-FUNCTION AND GRADIENT
                f[] += spsa.__cfd_0.coefs[i] * fe
                g  .+= spsa.__cfd_1.coefs[i] .* fe .* e ./ h
            end
        end
        f[] /= spsa.n
        g  ./= spsa.n

        # First-order methods: descent is along gradient.
        p .= g

        # Enforce trust region.
        if spsa.trust_region < typemax(Float)
            norm = LinearAlgebra.norm(p)
            if norm > spsa.trust_region
                p .*= (spsa.trust_region / norm)
            end
        end

        # Update parameters.
        η = Streams.next!(spsa.η)
        x .-= η .* p

        return false
    end
end