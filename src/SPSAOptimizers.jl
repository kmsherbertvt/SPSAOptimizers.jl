module SPSAOptimizers
    const Float = Float64

    #= The `Serialization` sub-module designs a standardized interface
        for representing optimization states as JSON-serializable objects.

    This module is perhaps badly named,
        because there is a module in base Julia called `Serialization`
        which goes about a similar task in a completely incompatible way...
    Don't confuse them!
    =#
    include("serialization.jl")
        import .Serialization: serialize, deserialize

    #= Several different options in SPSA can be understood
        from the perspective of an unending sequence, or "stream", of numbers.

    In qiskit, these are implemented very naturally with generators.
    The `Streams` sub-module defines a (serializable) interface
        to achieve the same flexibility in the Julia code.
    =#
    include("streams.jl")
        import .Streams: StreamType
    include("streams/seriesstreams.jl")
        import .PowerStreams: PowerSeries
        import .ConstantStreams: ConstantSeries
    include("streams/randomstreams.jl")
        import .RandomStreams: BernoulliDistribution
    include("streams/dictstreams.jl")
        import .DictStreams: IntDictStream

    #= Second-order methods require some representation of the Hessian,
        and some rules for updating it with additional information.
    The `Hessians` sub-module defines the interface needed for SPSA.
    But the standard implementation just uses the so-called `TrajectoryHessian`,
        which takes "the" Hessian used to pre-condition the gradient
        as the average of all Hessians measured in all optimization steps.
    =#
    include("hessians.jl")
        import .Hessians: HessianType
        import .TrajectoryHessians: TrajectoryHessian

    #= The `Optimizers` sub-module is where we define
        the fully-customizable interface for running optimizations.

    Actual optimization algorithms are implemented within the `optimizers` directory.
    =#
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

    #= The `QiskitInterface` sub-module is meant to emulate
        the (not quite as customizable) SPSA interface in qiskit
        as closely as possible.

    If the goal is to do practice runs for IBMQ,
        probably you might as well use this module instead of `Optimizations`.

    =#
    include("qiskit_interface.jl")
        import .QiskitInterface

end
