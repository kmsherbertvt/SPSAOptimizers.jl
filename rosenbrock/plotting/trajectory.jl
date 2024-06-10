"""
Show trajectories in parameter space.
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
        plot!(plt, data.x, data.y;
            markershape=:circle, markerstrokewidth=0, linewidth = 2, label=false,
            kwargs...,
        )
    end
end