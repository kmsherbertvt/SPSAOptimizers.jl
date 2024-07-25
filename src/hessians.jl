module Hessians

    """
    An internal representation of a "tracked" Hessian,
        e.g. the average of Hessians measured at each iteration in the trajectory.
    """
    abstract type HessianType{F} end

    """
        update!(H::HessianType{F}, Hm::AbstractMatrix{F})

    Update the "tracked" Hessian `H` with the "measured" Hessian `Hm`.
    """
    function update! end

    """
        regularized(H::HessianType{F})

    A bit of post-processing to construct a positive semi-definite version of `H`.
    """
    function regularized end

end


module TrajectoryHessians
    import ..Float
    import ..Serialization
    import ..Hessians

    import LinearAlgebra

    """
        TrajectoryHessian(bias, H, k)

    Hessian is average of Hessians from each iteration in trajectory.

    # Parameters
    - bias::Float - the regularization bias,
        i.e. the iotum of an identity matrix to add
        when computing a positive semi-definite version of this matrix
    - H::Matrix{Float} - the average Hessian over all `k` iterations
    - k::Ref{Int} - the number of iterations over which `H` is averaged

        TrajectoryHessian(L; bias)

    Convenience constructor which initializes `H` to an LxL matrix,
        and `k` to 0.
    The `bias` defaults to 0.01 but can be overridden.

    Note that we initialize `H` to identity, but this does not matter because,
        with k=0, it will be weighted by 0 in the first update!

    """
    struct TrajectoryHessian <: Hessians.HessianType{Float}
        bias::Float
        H::Matrix{Float}
        k::Ref{Int}
    end

    TrajectoryHessian(L::Int; bias::Float=0.01) = TrajectoryHessian(
        bias,
        Matrix{Float}(LinearAlgebra.I, L, L),
        Ref(0),
    )

    Serialization.__register__(TrajectoryHessian)

    Serialization.__data__(H::TrajectoryHessian) = (
        bias = H.bias,
        H = H.H,   # TODO: Matrices aren't directly serializable, are they? I expect we'll need to write H as a list of lists.
        k = H.k[],
    )

    Serialization.init(
        ::Type{TrajectoryHessian},
        data,
    ) = TrajectoryHessian(size(data.H, 1); bias=data.bias)
        # TODO: Probably size(., 1) -> length(.), if data.H is list of lists as I expect

    Serialization.load!(H::TrajectoryHessian, data) = (
        H.H  .= data.H; # TODO: Pry needs a loop, if data.H is list of lists as I expect
        H.k[] = data.k;
        H
    )

    Serialization.reset!(H::TrajectoryHessian) = (
        H.H  .= one(H.H);
        H.k[] = 0;
        H
    )

    Hessians.update!(H::TrajectoryHessian, Hu::AbstractMatrix{Float}) = (
        H.H   .*= H.k[];    # AVERAGE → SUM
        H.H   .+= Hu;       # EXTEND SUM
        H.k[]  += 1;        # UPDATE TOTAL COUNT
        H.H   ./= H.k[];    # AVERAGE ← SUM
        H
    )

    Hessians.regularized(H::TrajectoryHessian) = (
        sqrt(H.H * H.H') .+ H.bias .* one(H.H)
    )


end