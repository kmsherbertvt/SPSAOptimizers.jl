"""
"""
module FirstOrderOptimizers
    import ..Float
    import ..Serialization
    import ..Optimizers

    import Parameters: @with_kw

    import FiniteDifferences: FiniteDifferenceMethod, central_fdm
    import ..ConstantSeries

    """
    """
    @with_kw struct SPSA <: Optimizers.OptimizerType{Float}
        η::Streams.StreamType{Float}
        h::Streams.StreamType{Float}
        e::Streams.StreamType{Vector{Float}}
        p::Int = 2
        n::Int = 1
        trust_region::Float = typemax(Float)

        finite_difference = central_fdm(p,1)
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
    function iterate!(
        spsa::SPSA, x, fn;
        nfev::Ref{Int}=Ref(0),
        f::Ref{Float}=Ref(zero(Float)),
        g::Vector{Float}=Vector{Float}(undef, length(x)),
    )
        # Stochastically estimate the function and its gradient at this point.
        h = Streams.next!(spsa.h)
        evaluate!(f, g, x, fn, h, cfd, n)

        # Enforce trust region.
        if spsa.trust_region < typemax(Float)
            norm = norm(g)
            if norm > spsa.trust_region
                g .*= (spsa.trust_region / norm)
            end
        end

        # Update parameter
        η = Streams.next!(spsa.η)
        x .-= η .* g

        return false
    end

    """
    """
    function evaluate!(f, g, x, fn, h, cfd, n)

        # This header won't quite work; we need to sample e!

        #= TODO: The fun part. =#

    end




end