#=



        if self.second_order:
            points += [x + eps * (delta1 + delta2), x + eps * (-delta1 + delta2)]
            self._nfev += 2

        # batch evaluate the points (if possible)
        values = _batch_evaluate(loss, points, self._max_evals_grouped)

        plus = values[0]
        minus = values[1]
        gradient_sample = (plus - minus) / (2 * eps) * delta1

        hessian_sample = None
        if self.second_order:
            diff = (values[2] - plus) - (values[3] - minus)
            diff /= 2 * eps**2

            rank_one = np.outer(delta1, delta2)
            hessian_sample = diff * (rank_one + rank_one.T) / 2

----------------------------------------------------------------------

So it looks like this `delta2` is supposed to be some kind of forward finite difference of the gradient vector itself. My instinct tells me this is quite redundant and that a p+1,q+1 -order finite difference with the same direction is quite sufficient (and fits seamlessly into the code I've already written. Spall has directly contradicted my instinct in the past, so I'd best read his actual paper, but for now we'll do things my way.

Trouble is, I've no idea what to make of this `rank_one` object.
Oh, oh. We're trying to construct the matrix d^2 f / dx1 dx2, i.e. the Hessian itself. Each index of delta1 serves as dx1 and " " delta2 " dx2. The stochastic approach is of course not symmetric, so `(rank_one + rank_one.T) / 2` symmetrizes it, making it a plausible first guess at what the real Hessian might actually look like.

Okay, I get it now. This isn't quasi-Newton. This is bonafide Newton, albeit with simultaneous stochastic perturbation + smoothing over iterations to minimize measurement overhead. But we are actually doing *something* for each element of the Hessian matrix, and the measurement overhead turns quadratic. Quadratic in p, rather than n, but quadratic.

I'm afraid it's a little unintuitive, working out precisely how to reconstruct the `diff` from p unique measurements on all p perturbations. I'm sure I can work it out if I must, but for now, I'm inclined to just do exactly as qiskit does (and I bet that's what Spall first did too), which is to use p=2, and moreover to use a *forward* finite difference for the second derivative, so that we can reuse the sample points from the gradient.

I'll be honest, I think that's super inelegant and arbitrary. We should definitely implement the arbitrary p version. But what I have in mind certainly _will not_ do the forward finite difference, so we'd best make something directly comparable to qiskit first.

Right, so, I envision three "second order" methods:
- QiskitSPSA2: qiskit's version, with the p=2 and second order forward finite difference.
- SPSA2: Generic p version, otherwise identical.
- QuasiSPSA2: Use the gradients themselves to construct an approximation of Hk, ala BFGS. Removes need for extra measurements and for any `solve`.
















    regularization: float | None = None,
            This is fine as an arg. Default to 0.1 I think?
    hessian_delay: int = 0,
            Rather, iterate! takes a bool keyword argument `use_Hk`; delay handled in `optimize!` loop.
    lse_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
            I'd prefer to obivate the need for a solver by building up `Hk` rather than `H`, but the default matrix solver `H \ g` would work fine, so I don't presently think this needs to be an arg.
    initial_hessian: np.ndarray | None = None,
            This is fine as an arg. Default to identity, of course.
            No wait, iterate! just takes current Hessian!


=#

"""
"""
module SmoothHessians
    import ..Float
    import ..Serialization

    import LinearAlgebra

    """
    """
    abstract type HessianType{F} end

    """
        update!(sH::SmoothHessian{F}, H::AbstractMatrix{F})
    """
    function update! end

    """
        regularized(sH::SmoothHessian{F})
    """
    function regularized end



    """
        Hessian is average of Hessians from each iteration in trajectory.
    """
    struct TrajectoryHessian <: HessianType{Float}
        H::Matrix{Float}
        k::Int
        bias::Float
    end

    TrajectoryHessian(L::Int, bias::Float=0.1) = TrajectoryHessian(
        Matrix{Float}(LinearAlgebra.I, L, L),
        0,
        bias,
    )

    Serialization.__register__(TrajectoryHessian)

    Serialization.__data__(sH::TrajectoryHessian) = (
        H = sH.H,   # TODO: Matrices aren't directly serializable, are they? I expect we'll need to write H as a list of lists.
        k = sH.k,
        bias = sH.bias,
    )

    Serialization.init(
        ::Type{TrajectoryHessian},
        data,
    ) = TrajectoryHessian(size(data.H, 1), bias=data.bias)
        # TODO: Probably size(., 1) -> length(.), if data.H is list of lists as I expect

    Serialization.load!(sH::TrajectoryHessian, data) = (
        sH.H .= data.H; # TODO: Pry needs a loop, if data.H is list of lists as I expect
        sH.k = data.k;
        sH
    )

    Serialization.reset!(sH::TrajectoryHessian) = (
        sH.H .= one(sH.H);
        sH.k = 0;
        sH
    )

    update!(sH::TrajectoryHessian, H::AbstractMatrix{Float}) = (
        sH.H .*= sH.k;          # AVERAGE → SUM
        sH.H .+= H;             # EXTEND SUM
        sH.k  += 1;             # UPDATE TOTAL COUNT
        sH.H ./= sH.k;          # AVERAGE ← SUM
        sH
    )

    regularized(sH::TrajectoryHessian) = (
        sqrt(sH.H * sH.H') .+ sH.bias .* one(sH.H)
    )


end

"""
"""
module SecondOrderOptimizers
    import ..Float
    import ..Serialization
    import ..Optimizers

    import ..SmoothHessians
    import ..Streams

    import Parameters: @with_kw

    import LinearAlgebra

    import ..TrajectoryHessian
    import ..BernoulliDistribution

    """
    """
    @with_kw struct QiskitSPSA2 <: Optimizers.OptimizerType{Float}
        sH::SmoothHessians.HessianType{Float}
        η::Streams.StreamType{Float}
        h::Streams.StreamType{Float}
        e::Streams.StreamType{Vector{Float}}
        n::Int = 1
        trust_region::Float = typemax(Float)
    end

    function QiskitSPSA2(
        L::Int,
        η::Streams.StreamType{Float},
        h::Streams.StreamType{Float};
        sH = nothing,
        e = nothing,
        n = 1,
        trust_region = typemax(Float),
    )
        d = QiskitSPSA2(
            sH = isnothing(sH) ? TrajectoryHessian(L) : sH,
            η = η,
            h = h,
            e = isnothing(e) ? BernoulliDistribution(L=L) : e,
            n = n,
            trust_region = trust_region,
        )
    end

    Serialization.__register__(SPSA2)

    Serialization.__data__(spsa::SPSA2) = (
        sH = Serialization.__data__(spsa.sH),
        η = Serialization.__data__(spsa.η),
        h = Serialization.__data__(spsa.h),
        e = Serialization.__data__(spsa.e),
        p = spsa.p,
        n = spsa.n,
        trust_region = spsa.trust_region,
        regularization = spsa.regularization,
    )

    Serialization.init(::Type{SPSA2}, data) = Serialization.reset!(SPSA2(
        sH = Serialization.deserialize(data.sH),
        η = Serialization.deserialize(data.η),
        h = Serialization.deserialize(η),
        e = Serialization.deserialize(η),
        p = data.p,
        n = data.η,
        trust_region = data.trust_region,
        regularization = data.regularization,
    ))

    Serialization.load!(spsa::SPSA2, data) = (
        Serialization.load!(spsa.sH, data.sH);
        Serialization.load!(spsa.η, data.η);
        Serialization.load!(spsa.h, data.h);
        Serialization.load!(spsa.e, data.e);
        spsa
    )

    Serialization.reset!(spsa::SPSA2) = (
        Serialization.reset!(spsa.sH);
        Serialization.reset!(spsa.η);
        Serialization.reset!(spsa.h);
        Serialization.reset!(spsa.e);
        spsa
    )

    """
    Calculate H and update __smoothed_H,
        but only actually use it if `use_H` is `true`.
    This allows the optimize! loop to introduace a "delay" of several iterations over which the Hessian estimate will be smoothed out, before trying to invert a matrix.
    """
    function Optimizers.iterate!(
        spsa::SPSA2, x, fn;
        nfev::Ref{Int}=Ref(0),
        f::Ref{Float}=Ref(zero(Float)),
        g::Vector{Float}=Vector{Float}(undef, length(x)),
        H::Matrix{Float}=Matrix{Float}(undef, length(x), length(x)),
        p::Vector{Float}=Vector{Float}(undef, length(x)),
        solve_H = true,
        __xe::Vector{Float}=Vector{Float}(undef, length(x)),
        __e1::Vector{Float}=Vector{Float}(undef, length(x)),
    )
        f[] = 0
        g  .= 0
        H  .= 0

        # Stochastically estimate the function, gradient, and Hessian.
        h = Streams.next!(spsa.h)
        for _ in 1:spsa.n
            __e1 .= Streams.next!(spsa.e)
            e2 = Streams.next!(spsa.e)

            wt_H = zero(Float)

            # FORWARD ON e1, FROZEN ON e2
            __xe .= x .+ h .* __e1
            fe = fn(__xe); nfev[] += 1
            f[] += fe / 4
                #= NOTE: Divide by four here (rather than two) because qiskit
                    includes second-order measurements in the average.
                    I don't think it should... :/ =#
            g  .+= fe .* e ./ 2h
            wt_H -= fe / 2h^2

            # BACKWARD ON e1, FROZEN ON e2
            __xe .= x .- h .* __e1
            fe = fn(__xe); nfev[] += 1
            f[] += fe / 4
            g  .-= fe .* e ./ 2h
            wt_H += fe / 2h^2

            # FORWARD ON e1, FORWARD ON e2
            __xe .= x .+ h .* (__e1 .+ e2)
            fe = fn(__xe); nfev[] += 1
            f[] += fe / 4
            wt_H += fe / 2h^2

            # BACKWARD ON e1, FORWARD ON e2
            __xe .= x .- h .* (__e1 .- e2)
            fe = fn(__xe); nfev[] += 1
            f[] += fe / 4
            wt_H -= fe / 2h^2

            # ADD THE RANK ONE TENSOR `wt_H * outer(e1 e2)`, symmetrized
            LinearAlgebra.mul!(H, __e1, e2', wt_H / 2, one(Float))
            LinearAlgebra.mul!(H, e2, __e1', wt_H / 2, one(Float))
        end
        f[] /= spsa.n
        g  ./= spsa.n
        H  ./= spsa.n

        # Update smoothed Hessian.
        SmoothHessians.update!(spsa.sH, H)

        # Combine Hessian and gradient to decide the actual descent.
        p .= solve_H ? SmoothHessians.regularized(spsa.sH) \ g : g
            #= NOTE: I feel like there is no way to avoid allocations here,
                so I'm not even trying. =#

        # Enforce trust region.
        if spsa.trust_region < typemax(Float)
            norm = LinearAlgebra.norm(p)
            if norm > spsa.trust_region
                p .*= (spsa.trust_region / norm)
            end
        end

        # Update parameters.
        η = Streams.next!(spsa.η)
        x .-= η .* p

        return false
    end
end