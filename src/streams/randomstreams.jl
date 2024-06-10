"""
"""
module RandomStreams
    import ..Float
    import ..Serialization
    import ..Streams

    import Random
    import Parameters: @with_kw


    """
    """
    @with_kw struct BernoulliDistribution <: Streams.StreamType{Vector{Float}}
        L::Int
        k::Int = L
        p::Float = one(Float)
        seed::UInt = 0

        __vector__::Vector{Float} = Vector{Float}(undef, L)
        __rng__::Random.Xoshiro = Random.Xoshiro(seed)
    end

    Serialization.__register__(BernoulliDistribution)

    Serialization.__data__(X::BernoulliDistribution) = (
        L = X.L,
        k = X.k,
        p = X.p,
        seed = X.seed,
        s0 = X.__rng__.s0,
        s1 = X.__rng__.s1,
        s2 = X.__rng__.s2,
        s3 = X.__rng__.s3,
    )

    Serialization.init(::Type{BernoulliDistribution}, data) = BernoulliDistribution(
        L = data.L,
        k = data.k,
        p = data.p,
        seed = data.seed,
    )

    Serialization.load!(X::BernoulliDistribution, data) = (
        X.__rng__.s0 = data.s0;
        X.__rng__.s1 = data.s1;
        X.__rng__.s2 = data.s2;
        X.__rng__.s3 = data.s3;
        X
    )

    Serialization.reset!(X::BernoulliDistribution) = (
        rng = Random.Xoshiro(X.seed);
        X.__rng__.s0 = rng.s0;
        X.__rng__.s1 = rng.s1;
        X.__rng__.s2 = rng.s2;
        X.__rng__.s3 = rng.s3;
        X
    )



    """
    NOTE: r=1.0 is defined to give 0, irrespective of X.p
    """
    function sample(X::BernoulliDistribution, r)
        r -= X.p/2; r < 0 && return -one(Float)
        r -= X.p/2; r < 0 && return +one(Float)
        return zero(Float)
    end

    """
    """
    function Streams.next!(X::BernoulliDistribution)
        X.__vector__ .= 1
        Random.rand!(X.__rng__, @view(X.__vector__[1:X.k]))
        X.k < X.L && Random.shuffle!(X.__rng__, X.__vector__)
        for i in 1:X.L
            X.__vector__[i] = sample(X, X.__vector__[i])
        end
        return X.__vector__
    end

end