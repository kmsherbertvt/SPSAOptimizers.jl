"""
f, g vs iter, with optional twinned axis for #f also.
"""
module ConvergencePlots
    using Plots

    function init(; log=true, nfev=false, kwargs...)
        plt = plot(;
            xlabel = "Iterations",
            xlims = (0,Inf),
            ylabel = "Cost Function",
            ylims = log ? [1e-8, 1e2] : [0.0, 2.0],
            yticks = log ? 10.0 .^ (-8:2:2) : 0.0:0.2:2.0,
            yscale = log ? :log10 : :linear,
            kwargs...,
        )

        nfev || return plt

        twin = twinx(plt)
        plot!(twin;
            xlims = (0,Inf),
            ylabel = "# Function Evaluations",
            ylims = (0,Inf),
        )

        return plt, twin
    end

    function add!(plt, data; include_g=false, kwargs...)
        plot!(plt, data.f;
            linestyle=:solid, linewidth = 3, label=false,
            kwargs...,
        )

        include_g && plot!(plt, data.g;
            kwargs...,
            linestyle=:dot, linewidth = 1, label=false,
        )
    end

    function add!(plt::Tuple, data; kwargs...)
        add!(plt[1], data; kwargs...)
        plot!(plt[2], data.nfev;
            kwargs...,
            linestyle=:dash, linewidth = 1, label=false,
        )
    end
end
