module MatrixPlots
    import DataFrames
    import Plots

    function make_plot(df,
        xfields, yfields, zfield;
        zlims=[0.0,1.0],
        xfocus=first(xfields),
        yfocus=first(yfields),
    )
        # FOR CONVENIENCE, `xfields` AND `yfields` MAY INCLUDE FOCI
        xfields = setdiff(xfields, [xfocus, yfocus])
        yfields = setdiff(yfields, [xfocus, yfocus])
        allxfields = [xfocus; xfields]
        allyfields = [yfocus; yfields]
        allfields = [allxfields; allyfields]

        #= We're going to just implicitly assume that
            {xfocus, xfields..., yfocus, yfields...} forms a unique multi-key.
        That is, we assume that there is only one row
            with any given choice of values for the multi-key.
        Nothing will break if that isn't the case;
            it just means there's no guarantee about what gets plotted. =#

        multiindex = make_multiindex(df, allfields)
        bases = make_bases(multiindex, allfields)
        # bases = DataFrames.combine(multiindex, allfields .=> maximum; renamecols=false)
        indices = make_cartesianindex(multiindex, bases, allxfields, allyfields)

        z = fill(zlims[1], maximum(indices[!,:j]), maximum(indices[!,:i]))
        for (i, row) in enumerate(eachrow(indices))
            z[row[:j], row[:i]] = df[i,zfield]
        end
        # TODO: This is a hack to tide us over 'til we get the colorbar right.
        z[z .<= zlims[1]] .= [zlims[1]]; z[1,1] = zlims[1]
        z[z .>= zlims[2]] .= [zlims[2]]; z[end,end] = zlims[2]

        xlocs = cumprod(bases[1,field] for field in allxfields)
        ylocs = cumprod(bases[1,field] for field in allyfields)
        #= TODO: I suppose we could filter out any where base=1,
            but let's assume the user already has, for now. =#

        plt = Plots.heatmap(z;
            aspect_ratio=:equal, #framestyle=:origin,
            xlims=0.5.+[0,maximum(xlocs)], ylims=0.5.+[0,maximum(ylocs)],
            xticks=(xlocs, allxfields), yticks=(ylocs, allyfields),
            colorbar=false,
        )

        lw = 1
        for interval in xlocs
            Plots.vline!(plt, 0.5 .+ (0:interval:maximum(xlocs));
                label=false, color=:black, lw=lw/2)
            lw += 1
        end
        lw = 1
        for interval in ylocs
            Plots.hline!(plt, 0.5 .+ (0:interval:maximum(ylocs));
                label=false, color=:black, lw=lw/2)
            lw += 1
        end
    end



    function make_multiindex(df, fields)
        uniques = NamedTuple(field => sort(unique(df[!,field])) for field in fields)

        multiindex = DataFrames.DataFrame()
        for field in fields
            multiindex[!,field] = indexin(df[!,field], uniques[field])
        end

        return multiindex
    end

    function make_bases(multiindex, fields)
        return DataFrames.combine(multiindex, fields .=> maximum; renamecols=false)
    end

    function calculate_index(multiindex_row, bases, fields)
        ix = 1
        place = 1
        for field in fields
            ix += place * (multiindex_row[field] - 1)
            place *= bases[1,field]
        end
        return ix
    end

    function make_cartesianindex(multiindex, bases, ifields, jfields)
        get_i(row) = calculate_index(row, bases, ifields)
        get_j(row) = calculate_index(row, bases, jfields)

        indices = DataFrames.DataFrame()
        indices[!,:i] = [get_i(row) for row in eachrow(multiindex)]
        indices[!,:j] = [get_j(row) for row in eachrow(multiindex)]

        return indices
    end
end