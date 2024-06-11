"""
Show trajectories in parameter space.

It shouldn't throw an error, but the plot probably only makes sense for N=2 parameters.

"""
module TrajectoryPlots
    using Plots

    function init(fn; xlims=[-0.1,1.1], ylims=[-0.1,1.1], kwargs...)
        x = xlims[1]:.001:xlims[2]
        y = ylims[1]:.001:ylims[2]

        return contourf(
            x, y, (x, y) -> fn([x,y]);
            xlims=xlims,
            ylims=ylims,
            aspect_ratio=:equal,
            levels=[0, 10.0 .^ (-4:1:4)...],
            colorbar=false,
            kwargs...,
        )
    end

    function add!(plt, data; kwargs...)
        plot!(plt, data.x[:,1], data.x[:,2];
            markershape=:circle, markerstrokewidth=0, linewidth = 2, label=false,
            kwargs...,
        )
    end
end