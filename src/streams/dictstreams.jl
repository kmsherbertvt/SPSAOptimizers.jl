"""
"""
module DictStreams
    import ..Int
    import ..Serialization
    import ..Streams

    import Parameters: @with_kw

    """
    """
    @with_kw struct IntDictStream <: Streams.StreamType{Int}
        dict::Dict{Int,Int} = Dict{Int,Int}()
        default::Int = zero(Int)
        k::Ref{Int} = Ref(zero(Int))
    end

    Serialization.__register__(IntDictStream)

    Serialization.__data__(D::IntDictStream) = (
        dict = D.dict,
        default = D.default,
        k = D.k[],
    )

    Serialization.init(::Type{IntDictStream}, data) = IntDictStream(
        dict = data.dict,
        default = data.default,
    )

    Serialization.load!(D::IntDictStream, data) = (
        D.k[] = data.k;
        D
    )

    Serialization.reset!(D::IntDictStream) = (
        D.k[] = 0;
        D
    )

    """
    """
    function Streams.next!(D::IntDictStream)
        D.k[] += 1
        return get(D.dict, D.k[]-1, D.default)
    end

end