var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SPSAOptimizers","category":"page"},{"location":"#SPSAOptimizers","page":"Home","title":"SPSAOptimizers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SPSAOptimizers.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [\n    SPSAOptimizers,\n    Serialization,\n    Streams,\n    PowerStreams,\n    ConstantStreams,\n    RandomStreams,\n    DictStreams,\n    Hessians,\n    TrajectoryHessians,\n    Optimizers,\n    FirstOrderOptimizers,\n    SPSA1s,\n    SecondOrderOptimizers,\n    SPSA2s,\n    QiskitInterface,\n]","category":"page"},{"location":"#SPSAOptimizers.Serialization","page":"Home","title":"SPSAOptimizers.Serialization","text":"\n\n\n\n","category":"module"},{"location":"#SPSAOptimizers.Serialization.__data__","page":"Home","title":"SPSAOptimizers.Serialization.__data__","text":"\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.Serialization.deserialize-Tuple{Any}","page":"Home","title":"SPSAOptimizers.Serialization.deserialize","text":"deserialize(json)\n\nConvert a JSON-serializable NamedTuple into an object.\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.Serialization.init","page":"Home","title":"SPSAOptimizers.Serialization.init","text":"init(::Type{<:Serializable}, data)\n\nIgnore mutable attributes in data; just set state-independent values.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.Serialization.load!","page":"Home","title":"SPSAOptimizers.Serialization.load!","text":"load!(::Serializable, data)\n\nIgnore immutable attributes in data; just change state-dependent values.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.Serialization.reset!","page":"Home","title":"SPSAOptimizers.Serialization.reset!","text":"reset!(::Serializable)\n\nRestore mutable attributes to their initial values.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.Serialization.serialize-Tuple{Any}","page":"Home","title":"SPSAOptimizers.Serialization.serialize","text":"serialize(object)\n\nConvert an object into a JSON-serializable NamedTuple.\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.Streams.StreamType","page":"Home","title":"SPSAOptimizers.Streams.StreamType","text":"StreamType{T}\n\nConcrete StreamTypes should implement the interface for serialization     (data, init, load!, reset!, along with calling register)     in addition to the next! function, which returns an object of type T.\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.Streams.next!","page":"Home","title":"SPSAOptimizers.Streams.next!","text":"next!(stream::StreamType{T})\n\nReturns the next element of type T from the given stream object.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.PowerStreams","page":"Home","title":"SPSAOptimizers.PowerStreams","text":"Guidelines for selecting gain sequences:\n\nLet η[k] = η0 / (A + k + 1)^α     h[k] = h0 / (k + 1)^γ\n\nAsymptotically optimal: α=1, γ=1/6\nLowest allowable for theoretical convergence: α=0.602, γ=0.101\nEmpirically, the smaller the better, but maybe for large problems one should transition to the larger.\nSet h0 to the standard deviation of the noise in f.\nSet A to 10% of the maximum number of iterations.\nSet a0 such that a[0] * |g(x0)| gives a desirable \"change in magnitude\" of x.\n\n\n\n\n\n","category":"module"},{"location":"#SPSAOptimizers.PowerStreams.PowerSeries","page":"Home","title":"SPSAOptimizers.PowerStreams.PowerSeries","text":"PowerSeries(a0, γ, A)\n\nYield from the sequence ak = a_0  (A + k + 1)^γ.\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.ConstantStreams.ConstantSeries","page":"Home","title":"SPSAOptimizers.ConstantStreams.ConstantSeries","text":"ConstantSeries(C)\n\nYield the number C, every time.\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.Streams.next!-Tuple{SPSAOptimizers.ConstantStreams.ConstantSeries}","page":"Home","title":"SPSAOptimizers.Streams.next!","text":"\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.RandomStreams","page":"Home","title":"SPSAOptimizers.RandomStreams","text":"\n\n\n\n","category":"module"},{"location":"#SPSAOptimizers.RandomStreams.BernoulliDistribution","page":"Home","title":"SPSAOptimizers.RandomStreams.BernoulliDistribution","text":"BernoulliDistribution(L, k, p, seed)\n\nA Bernoulli distribution yielding a vector of +/-1.\n\nParameters\n\nL::Int, the length of the vector to generate\nk::Int=L, the number of elements in the vector to randomize   If an element isn't randomized, it is set to 0.   Which elements to randomize are...selected randomly. :)\np::Int=1.0, the probability of randomizing any given element in the vector\n\nNOTE: k and p are different and mutually exclusive ways     to let 0's sneak into the Bernoulli distribution. The defaults are just \"no 0's allowed\",         i.e. an actual Bernoulli distribution.\n\nseed::Int, the random seed defining the pseudorandom stream\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.RandomStreams.sample-Tuple{SPSAOptimizers.RandomStreams.BernoulliDistribution, Any}","page":"Home","title":"SPSAOptimizers.RandomStreams.sample","text":"NOTE: r=1.0 is defined to give 0, irrespective of X.p\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.DictStreams","page":"Home","title":"SPSAOptimizers.DictStreams","text":"\n\n\n\n","category":"module"},{"location":"#SPSAOptimizers.DictStreams.IntDictStream","page":"Home","title":"SPSAOptimizers.DictStreams.IntDictStream","text":"IntDictStream(dict, default)\n\nA customized stream specifying the output for any given specific iteration in dict,     or otherwise a fixed constant default.\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.Hessians.HessianType","page":"Home","title":"SPSAOptimizers.Hessians.HessianType","text":"An internal representation of a \"tracked\" Hessian,     e.g. the average of Hessians measured at each iteration in the trajectory.\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.Hessians.regularized","page":"Home","title":"SPSAOptimizers.Hessians.regularized","text":"regularized(H::HessianType{F})\n\nA bit of post-processing to construct a positive semi-definite version of H.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.Hessians.update!","page":"Home","title":"SPSAOptimizers.Hessians.update!","text":"update!(H::HessianType{F}, Hm::AbstractMatrix{F})\n\nUpdate the \"tracked\" Hessian H with the \"measured\" Hessian Hm.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.TrajectoryHessians.TrajectoryHessian","page":"Home","title":"SPSAOptimizers.TrajectoryHessians.TrajectoryHessian","text":"TrajectoryHessian(bias, H, k)\n\nHessian is average of Hessians from each iteration in trajectory.\n\nParameters\n\nbias::Float - the regularization bias,   i.e. the iotum of an identity matrix to add   when computing a positive semi-definite version of this matrix\nH::Matrix{Float} - the average Hessian over all k iterations\nk::Ref{Int} - the number of iterations over which H is averaged\nTrajectoryHessian(L; bias)\n\nConvenience constructor which initializes H to an LxL matrix,     and k to 0. The bias defaults to 0.01 but can be overridden.\n\nNote that we initialize H to identity, but this does not matter because,     with k=0, it will be weighted by 0 in the first update!\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.Optimizers.OptimizerType","page":"Home","title":"SPSAOptimizers.Optimizers.OptimizerType","text":"Used for dispatch to different optimization algorithms.\n\nAlgorithms are organized by first-order, second-order, and so on. (Well, just those two for now,     but I feel like there could be a \"quasi-second order\" someday.)\n\nConstructors are documented with concrete types (e.g. SPSA1),     but details on optimization options,     record schematics, and iterate! keyword arguments     are found in the intermediate order types, e.g. FirstOrderOptimizer.\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.Optimizers.Record","page":"Home","title":"SPSAOptimizers.Optimizers.Record","text":"Record(optimizer, L)\n\nInitialize a mutable object giving the status of an L-dimensional optimization.\n\nThe object is a NamedTuple of vectors and references,     whose precise schema depends on the type of optimizer.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.Optimizers.Trace-Tuple{}","page":"Home","title":"SPSAOptimizers.Optimizers.Trace","text":"Trace()\n\nInitialize a vector of Records, giving a trace of an optimization.\n\nSimply allows for the more intuitive trace initialization, Trace(). Asking people to type NamedTuple[] just sounds so...pretentious.\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.Optimizers.averagerecords!-Tuple{Any, Any}","page":"Home","title":"SPSAOptimizers.Optimizers.averagerecords!","text":"averagerecords!(archive, records)\n\nLike copyrecord!, but stores an average of many records in archive.\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.Optimizers.copyrecord!-Tuple{Any, Any}","page":"Home","title":"SPSAOptimizers.Optimizers.copyrecord!","text":"copyrecord!(archive, record)\n\nCopy all values from the record Record to the archive Record.\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.Optimizers.iterate!","page":"Home","title":"SPSAOptimizers.Optimizers.iterate!","text":"iterate!(optimizer, fn, x; kwargs...)\n\nAdvance by one step in an optimization routine, updating x in place.\n\nIn addition to updating x, this will update the state of optimizer.\n\nParameters\n\noptimizer::OptimizerType{F} - defines the optimization algorithm\nfn - the loss-function, such that fn(x) returns a number of type F\nx::AbstractVector{F} - the best guess so far\n\nKeyword arguments depend on the type of optimizer.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.Optimizers.optimize!","page":"Home","title":"SPSAOptimizers.Optimizers.optimize!","text":"optimize!(optimizer, fn, x0; kwargs...)\n\nRun an optimization routine to convergence, updating x in place.\n\nIn addition to updating x, this will update the state of optimizer.\n\nParameters\n\noptimizer::OptimizerType{F} - defines the optimization algorithm\nfn - the loss-function, such that fn(x) returns a number of type F\nx::AbstractVector{F} - the best guess so far\n\nKeyword Arguments\n\nmaxiter::Int - the max number of iterations to attempt, successful or not\ncallback - function called at each iteration, successful or not:\ncallback(optimizer, iterate)\nwhere iterate is a record of the proposed step. The callback is called immediately prior   to deciding whether to accept or reject the step,   and the step is always rejected if the callback returns true.\nrecord, average_last, trace, tracefields - see Output section below\n\nAdditional keyword arguments may be available     depending on the type of optimizer.\n\nOutput\n\nA record (a NamedTuple with schema specified by the optimizer)     is returned at the end. If you provide an integer to average_last,     that record is an average of the last so many (successful) iterations.\n\nIf you provide an object to the record keyword argument,     that object will be the one used for the final return output. More importantly, this same object is used to keep track of the     last successful iteration,     so it will have meaningful data if something (e.g. keyboard interrupt)     disrupts the algorithm. Note however that if terminated prematurely,     record will only represent the last successful iteration,     rather than an average, even if average_last is provided.\n\nYou can also provide a trace,     which is a list of records of every single step. Provide a collection of field names (as symbols) to specify which fields     should be saved in the trace. The default is the collection of all scalar attributes. Set it to an empty collection to trace every possible attribute.\n\n\n\n\n\n","category":"function"},{"location":"#SPSAOptimizers.Optimizers.trace!-Tuple{Any, Any, Vararg{Any}}","page":"Home","title":"SPSAOptimizers.Optimizers.trace!","text":"trace!(trace, record, fields...)\n\nCopies a record and adds the copy to the trace.\n\nOptionally, list specific fields - all others will be omitted from the copy.\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.Optimizers.trajectory-Tuple{Any, Any}","page":"Home","title":"SPSAOptimizers.Optimizers.trajectory","text":"trajectory(trace, field)\n\nExtract the values of a particular field from each record in a trace,     and return as a vector (if field names a reference)     or a matrix, where each column is the vector stored in one record.\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.FirstOrderOptimizers.FirstOrderOptimizer","page":"Home","title":"SPSAOptimizers.FirstOrderOptimizers.FirstOrderOptimizer","text":"FirstOrderOptimizer{F}\n\nSuper-type for first-order optimizers. (I'm pretty sure there's just the one, SPSA1.) Constructors are documented with the concrete type(s?),     but find details on optimization options,     record schematics, and iterate! keyword arguments here:\n\nRecord Schematic\n\nRecord(::FirstOrderOptimizer{F}, L::Int)\n\nConstructs a NamedTuple with fields\n\nx::Vector{F} of length L, the current parameters\nf::Ref{F}, the current function value\ng::Vector{F} of length L, the current gradient estimate\np::Vector{F} of length L, the current parameter step (based on gradient)\nxp::Vector{F} of length L, the proposed parameters for the next iteration\nfp::Ref{F}, the function value at the next iteration\nnfev::Ref{Int}, the number of function evaluations to date\ntime::Ref{F}, the algorithm time to date\nbytes::Ref{F}, the memory consumption to date\n\nAdditional optimize! Keyword Arguments\n\ntrust::F=Inf, the maximum allowable norm of p\nIf p has a larger norm, it gets clipped to trust.\ntolerance::F=Inf, the amount by which fp may exceed f without rejecting the step\n\niterate! Interface\n\niterate!(\n    optimizer, fn, x;\n    # EXTRA OPTIONS\n    trust,\n    # TRACEABLES\n    f, g, p, xp, nfev,\n    # WORK ARRAYS\n    __xe\n)\n\noptimizer, fn, x: the mandatory positional arguments, see iterate!\ntrust: see above\nf, g, p, xp, nfev: when provided, stores meaningful outputs   (when called through optimize!,       these are provided from the record of the last successful iteration).\n__xe: a work variable matching the dimensions of x,   used to store the stochastically perturbed parameters for finite difference\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.SPSA1s.SPSA1","page":"Home","title":"SPSAOptimizers.SPSA1s.SPSA1","text":"SPSA1{P}(η, h, e, n)\n\nThe first-order SPSA optimizer.\n\nType Parameters\n\nP::Int, the order of finite difference, typically 2\n\nParameters\n\nη - the float stream for step-length at each iteration\nh - the float stream for finite-difference perturbation at each iteration\ne - the vector stream for each finite difference perturbation direction   This also determines the dimensions in the parameter vector.\nn - the int stream for number of times to sample the gradient at each iteration\n\nSPSA1(L; P, η, h, e, n)\n\nWith this constructor, you need only provide the number of dimensions L,     and all the above parameters will be filled in by sensible defaults     (but you can override any of them with the kwarg).\n\nAdditionally, you may use the η and h kwargs to specify tuples     defining a power series.\n\nDefaults\n\nP: 2, of course - central finite difference\nη: a power series with a0 = 0.2, alpha=0.602, and A = 0\nh: a power series with c0 = 0.2 and gamma=0.101\ne: a fair coin-toss between +/-1 for each dimension\nn: one sample for every iteration\n\nTuple Interface\n\nIf you don't feel like manually constructing the PowerStream object,     you can just pass a tuple directly here.\n\nη: (a0, alpha, A) all floats (even though A is semantically an integer)\nh: (c0, gamma) all floats (note A is always 0 for the h stream)\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.SecondOrderOptimizers.SecondOrderOptimizer","page":"Home","title":"SPSAOptimizers.SecondOrderOptimizers.SecondOrderOptimizer","text":"SecondOrderOptimizer{F}\n\nSuper-type for first-order optimizers. (Just SPSA2 for now but I have visions of how to enhance.     They probably aren't worth the effort, though.) Constructors are documented with the concrete type(s?),     but find details on optimization options,     record schematics, and iterate! keyword arguments here:\n\nRecord Schematic\n\nRecord(::FirstOrderOptimizer{F}, L::Int)\n\nConstructs a NamedTuple with fields\n\nx::Vector{F} of length L, the current parameters\nf::Ref{F}, the current function value\ng::Vector{F} of length L, the current gradient estimate\nH::Matrix{F} of size(L,L), the current Hessian estimate\np::Vector{F} of length L, the current parameter step (based on gradient)\nxp::Vector{F} of length L, the proposed parameters for the next iteration\nfp::Ref{F}, the function value at the next iteration\nnfev::Ref{Int}, the number of function evaluations to date\ntime::Ref{F}, the algorithm time to date\nbytes::Ref{F}, the memory consumption to date\n\nAdditional optimize! Keyword Arguments\n\ntrust::F=Inf, the maximum allowable norm of p\nIf p has a larger norm, it gets clipped to trust.\ntolerance::F=Inf, the amount by which fp may exceed f without rejecting the step\nwarmup::Int=0, after how many iterations to begin refining the step with the Hessian   (rather than just using gradient directly)\nsolver, a function with header solver(H, g)   returning the \"preconditioned\" H^-1 g\n\niterate! Interface\n\niterate!(\n    optimizer, fn, x;\n    # EXTRA OPTIONS\n    trust, precondition, solver,\n    # TRACEABLES\n    f, g, H, p, xp, nfev,\n    # WORK ARRAYS\n    __xe,\n    __e1,\n)\n\noptimizer, fn, x: the mandatory positional arguments, see iterate!\ntrust, solver: see above\nprecondition: whether or not to use the Hessian in this iteration\nf, g, H, p, xp, nfev: when provided, stores meaningful outputs   (when called through optimize!,       these are provided from the record of the last successful iteration).\n__xe: a work variable matching the dimensions of x,   Used to store the stochastically perturbed parameters for finite difference\n__e1: a work variable matching the dimensions of x   Used to store one of the two perturbation vectors       needed for second order finite difference.   Note that the other one is stored within the stream object,       which is why this isn't needed at all in first-order.\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.SPSA2s.SPSA2","page":"Home","title":"SPSAOptimizers.SPSA2s.SPSA2","text":"SPSA2(H, η, h, e, n)\n\nThe second-order SPSA optimizer.\n\nParameters\n\nH - the Hessian object, updated at each iteration\nη - the float stream for step-length at each iteration\nh - the float stream for finite-difference perturbation at each iteration\ne - the vector stream for each finite difference perturbation direction   This also determines the dimensions in the parameter vector.\nn - the int stream for number of times to sample the gradient at each iteration\n\nSPSA2(L; H, η, h, e, n)\n\nWith this constructor, you need only provide the number of dimensions L,     and all the above parameters will be filled in by sensible defaults     (but you can override any of them with the kwarg).\n\nAdditionally, you may use the η and h kwargs to specify tuples     defining a power series.\n\nDefaults\n\nH: a fresh LxL TrajectoryHessian with regularization bias 0.01\nη: a power series with a0 = 0.2, alpha=0.602, and A = 0\nh: a power series with c0 = 0.2 and gamma=0.101\ne: a fair coin-toss between +/-1 for each dimension\nn: one sample for every iteration\n\nTuple Interface\n\nIf you don't feel like manually constructing the PowerStream object,     you can just pass a tuple directly here.\n\nη: (a0, alpha, A) all floats (even though A is semantically an integer)\nh: (c0, gamma) all floats (note A is always 0 for the h stream)\n\n\n\n\n\n","category":"type"},{"location":"#SPSAOptimizers.QiskitInterface","page":"Home","title":"SPSAOptimizers.QiskitInterface","text":"Using qiskit, one might do:\n\n```\n\nimport qiskit_algorithms\nimport qiskit_algorithms.optimizers as qopt\n\nqiskit_algorithms.utils.algorithm_globals.random_seed = seed\n\nget_eta = lambda: qopt.spsa.powerseries(eta=a0, power=alpha, offset=A)\nget_eps = lambda: qopt.spsa.powerseries(eta=c0, power=gamma)\n\nspsa = qopt.SPSA(\n    learning_rate = get_eta,\n    perturbation = get_eps,\n    **kwargs\n)\n\nresult = spsa.minimize(fn, x0)\n\n```\n\nWith this module, one could do this:\n\n```\nimport SPSAOptimizers.QiskitInterface as QI\n\nQI.seed!(seed)\n\nget_eta = QI.powerseries(; eta=a0, power=alpha, offset=A)\nget_eps = QI.powerseries(; eta=c0, power=gamma)\n\nspsa = QI.SPSA(length(x0);\n    learning_rate = get_eta,\n    perturbation = get_eps,\n    kwargs...\n)\n\nresult = QI.minimize(spsa, fn, x0)\n\n```\n\nNote that QI.SPSA needs a positional argument, not needed in Python. To make up for it, while the result object matches qiskit's interface exactly,     the spsa object contains attributes record and trace,     which, after the call to QI.minimize,     have some extra information unavailable to qiskit     (without writing your own callback function).\n\nTo avoid expensive and slightly arbitrary calibration steps,     this module treats learning_rate and perturbation as mandatory. Similarly, when the kwarg blocking is set to true,     the kwarg allowed_increase also becomes mandatory. While the qiskit implementation will perform a calibration     to select problem-informed defaults,     the calibration is not necessarily ... good. So you really should perform that calibration ahead of time     and then manually specify your parameters.\n\nInteresting note: qiskit's initial_hessian kwarg seems to be quite useless. That is, it sets the initial hessian,     but the initial hessian is, if I have read the code correctly,     always multiplied by zero before it is ever used. So. Don't use it. If you really want to, I've added an extra kwarg initial_hessian_weight     to the QI.SPSA interface, which can be a non-negative integer. For the record, the SPSAOptimizers package does permit an initial hessian,     but it would requires an extra hyperparameter for an \"initial weight\",     so let's just not use it with this QiskitInterface.\n\n\n\n\n\n","category":"module"},{"location":"#SPSAOptimizers.QiskitInterface.SPSA-Tuple{Int64}","page":"Home","title":"SPSAOptimizers.QiskitInterface.SPSA","text":"SPSA(L; kwargs...)\n\nInitialize an SPSA optimization. The interface is identical to qiskit's,     except that there is one required argument L giving the number of parameters,     and the learning_rate and perturbation arguments are mandatory.\n\n(Unlike in qiskit, the number of parameters has to be fixed ahead of time     to ensure the serializability of the random stream.)\n\nSee here for qiskit's interface:     https://docs.quantum.ibm.com/api/qiskit/0.28/qiskit.algorithms.optimizers.SPSA\n\n\n\n\n\n","category":"method"},{"location":"#SPSAOptimizers.QiskitInterface.minimize-Tuple{Any, Any, Any}","page":"Home","title":"SPSAOptimizers.QiskitInterface.minimize","text":"minimize(spsa, fn, x0)\n\nRun an optimization.\n\nParameters\n\nspsa - the result of QiskitInterface.SPSA\nfn - the function, fn(x) (where x is a vector of parameters)\nx0 - the initial guess\n\n\n\n\n\n","category":"method"}]
}
