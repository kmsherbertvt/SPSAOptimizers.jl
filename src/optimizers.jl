"""
"""
module Optimizers
    """
    """
    abstract type OptimizerType{F} end

    """

        iterate!(opt, x, fn; nfev::Ref{Int}, time::Ref{Float})

    """
    function iterate! end

    """
    """
    function optimize!(
        opt::OptimizerType{F}, x::AbstractVector{F}, fn;
        maxiter=100, nfev::Ref{Int}=Ref(0), time::Ref{F}=Ref(zero(F)),
    ) where {F}
        for iter in 1:maxiter
            iterate!(opt, x, fn; nfev=nfev) && break
        end
    end

    #=

    TODO: Special options in `optimize!` for `last_avg` and `blocking`,
        which can be handled in the outer loop.

    =#

end