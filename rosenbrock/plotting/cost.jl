"""
f vs #f
"""
module CostPlots
    using Plots

    init(; log=true, kwargs...) = plot(;
        xlabel = "# Function Evaluations",
        xlims = (0,Inf),
        ylabel = "Cost Function",
        ylims = log ? [1e-8, 1e2] : [0.0, 2.0],
        yticks = log ? 10.0 .^ (-8:2:2) : 0.0:0.2:2.0,
        yscale = log ? :log10 : :linear,
        kwargs...,
    )

    function add!(plt, data; include_g=false, kwargs...)
        plot!(plt, data.nfev, data.f;
            linestyle=:solid, linewidth = 3, label=false,
            kwargs...,
        )

        include_g && plot!(plt, data.nfev, data.g;
            kwargs...,
            linestyle=:dot, linewidth = 1, label=false,
        )
    end
end