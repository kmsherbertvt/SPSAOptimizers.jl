"""
"""
module PowerStreams
    import ..Float
    import ..Serialization
    import ..Streams

    import Parameters: @with_kw

    """
    """
    @with_kw struct PowerSeries <: Streams.StreamType{Float}
        a0::Float
        γ::Float
        A::Float = zero(Float)
        k::Ref{Int} = Ref(0)
    end

    Serialization.__register__(PowerSeries)

    Serialization.__data__(a::PowerSeries) = (
        a0 = a.a0,
        γ = a.γ,
        A = a.A,
        k = a.k[],
    )

    Serialization.init(::Type{PowerSeries}, data) = PowerSeries(
        a0 = data.a0,
        γ = data.γ,
        A = data.A
    )

    Serialization.load!(a::PowerSeries, data) = (
        a.k[] = data.k;
        a
    )

    Serialization.reset!(a::PowerSeries) = (
        a.k[] = 0;
        a
    )

    """
    """
    function Streams.next!(a::PowerSeries)
        a.k[] += 1
        return a.a0 / (a.A + a.k[])^a.γ
    end
end

"""
"""
module ConstantStreams
    import ..Float
    import ..Serialization
    import ..Streams

    struct ConstantSeries <: Streams.StreamType{Float}
        C::Float
    end

    Serialization.__register__(ConstantSeries)
    Serialization.__data__(a::ConstantSeries) = (C = a.C,)
    Serialization.init(::Type{ConstantSeries}, data) = ConstantSeries(data.C)
    Serialization.load!(a::ConstantSeries, data) = a
    Serialization.reset!(a::ConstantSeries) = a

    """
    """
    function Streams.next!(a::ConstantSeries)
        return a.C
    end
end