"""
"""
module Serialization
    const __REGISTRY__ = Dict{Symbol,Type}()
    __register__(type::Type) = __REGISTRY__[nameof(type)] = type

    """
    """
    function __data__ end

    """
        init(::Type{<:Serializable}, data)

    Ignore mutable attributes in data; just set state-independent values.

    """
    function init end

    """
        load!(::Serializable, data)

    Ignore immutable attributes in data; just change state-dependent values.

    """
    function load! end

    """
        reset!(::Serializable)

    Restore mutable attributes to their initial values.

    """
    function reset! end

    """
    """
    function serialize(object)
        return (
            type = String(nameof(typeof(object))),
            data = __data__(object),
        )
    end

    """
    """
    function deserialize(json)
        T = __REGISTRY__[Symbol(json.type)]
        object = init(T, json.data)
        load!(object, json.data)
        return object
    end
end