module Streams
    """
        StreamType{T}

    Concrete StreamTypes should implement the interface for serialization
        (__data__, init, load!, reset!, along with calling `register`)
        in addition to the `next!` function, which returns an object of type T.

    """
    abstract type StreamType{T} end

    """
        next!(stream::StreamType{T})

    Returns the next element of type T from the given stream object.

    """
    function next! end
end