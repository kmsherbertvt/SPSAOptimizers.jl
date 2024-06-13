module SPSAOptimizers
    const Float = Float64

    include("serialization.jl")
        import .Serialization: serialize, deserialize

    include("streams.jl")
        import .Streams: StreamType
    include("streams/seriesstreams.jl")
        import .PowerStreams: PowerSeries
        import .ConstantStreams: ConstantSeries
    include("streams/randomstreams.jl")
        import .RandomStreams: BernoulliDistribution
    include("streams/dictstreams.jl")
        import .DictStreams: IntDictStream

    include("hessians.jl")
        import .Hessians: HessianType
        import .TrajectoryHessians: TrajectoryHessian

    include("optimizers.jl")
        import .Optimizers: OptimizerType, iterate!, optimize!
        import .Optimizers: Record
        import .Optimizers: Trace, trajectory
    include("optimizers/firstorder.jl")
        import .FirstOrderOptimizers: FirstOrderOptimizer
        import .SPSA1s: SPSA1
    include("optimizers/secondorder.jl")
        import .SecondOrderOptimizers: SecondOrderOptimizer
        import .SPSA2s: SPSA2

    include("qiskit_interface.jl")
        import .QiskitInterface

end
