module SPSAOptimizers
    const Float = Float64

    include("serialization.jl")
        import Serialization: serialize, deserialize

    include("streams.jl")
        import Streams: StreamType
    include("streams/seriesstreams.jl")
        import PowerStreams: PowerSeries
        import ConstantStreams: ConstantSeries
    include("streams/randomstreams.jl")
        import RandomStreams: BernoulliDistribution

    include("optimizers.jl")
        import Optimizers: OptimizerType
    include("optimizers/firstorder.jl")
        import FirstOrderOptimzers: SPSA

end
