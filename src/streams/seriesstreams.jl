"""

Guidelines for selecting gain sequences:

Let η[k] = η0 / (A + k + 1)^α
    h[k] = h0 / (k + 1)^γ

- Asymptotically optimal: α=1, γ=1/6
- Lowest allowable for theoretical convergence: α=0.602, γ=0.101
- Empirically, the smaller the better, but maybe for large problems one should transition to the larger.

- Set h0 to the standard deviation of the noise in f.
- Set A to 10% of the maximum number of iterations.
- Set a0 such that a[0] * |g(x0)| gives a desirable "change in magnitude" of x.

"""
module PowerStreams
    import ..Float
    import ..Serialization
    import ..Streams

    import Parameters: @with_kw

    """
        PowerSeries(a0, γ, A)

    Yield from the sequence ``a[k] = a_0 / (A + k + 1)^γ``.

    """
    @with_kw struct PowerSeries <: Streams.StreamType{Float}
        a0::Float = one(Float)
        γ::Float = one(Float)
        A::Float = zero(Float)
        k::Ref{Int} = Ref(0)
    end

    PowerSeries(a0::Float) = PowerSeries(a0=a0)
    PowerSeries(a0::Float, γ::Float) = PowerSeries(a0=a0, γ=γ)
    PowerSeries(a0::Float, γ::Float, A::Float) = PowerSeries(a0=a0, γ=γ, A=A)

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

    function Streams.next!(a::PowerSeries)
        a.k[] += 1
        return a.a0 / (a.A + a.k[])^a.γ
    end
end

module ConstantStreams
    import ..Float
    import ..Serialization
    import ..Streams

    """
        ConstantSeries(C)

    Yield the number C, every time.

    """
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