import Plots
include("plotting/matshow.jl")
    import .MatrixPlots

import DataFrames, CSV
# infile = "surveys/surveys/rosenbrock/zeropoint/L20_M20000_σ0.0.csv"
# df = CSV.read(infile, DataFrames.DataFrame)

# xfields = [:tolerance, :n]
# yfields = [:η0, :α, :ApK, :order]

# plt = MatrixPlots.make_plot(df, :η0, :α, xfields, yfields, :fp)





infile = "surveys/surveys/rosenbrock/zeropoint/PROTOYPE_L20_M20000_σ0.0.csv"
df = CSV.read(infile, DataFrames.DataFrame)

xfields = [:h0, :η0, :NpL, :n, :trust]
yfields = [:γ, :α, :ApK, :order, :tolerance]


xfields = [:η0, :h0, :n, :NpL, :order]
yfields = [:trust, :α, :γ, :ApK, :tolerance]

multiindex = MatrixPlots.make_multiindex(df, [xfields..., yfields...])
xbases = MatrixPlots.make_bases(multiindex, xfields)
ybases = MatrixPlots.make_bases(multiindex, yfields)

plt = MatrixPlots.make_plot(df, xfields, yfields, :minf)


# #=

# How shall we visualize all this data..?
# For one thing, we can just throw all the minf from a CSV onto a number line,
#     and get a feel for clusters.
# Maybe it is worth trying to use colors/shapes to distinguish the big three:
# - order
# - n

# Or at least order and n. Maybe color for order and shape for n.



# Then we need some way of systematically checking the effect of a particular parameter. Like, assign to each [key not including the target parameter] a vector of [minf for each choice of parameter]. Then take a difference matrix of that vector. Look for large values, to identify the choice of hyperparameters for which the target parameter makes much difference. Sure why not, take the LARGEST value in the difference matrix as a scalar measure of "impact". (DataFrames may refer to this as the `range`..? ;)

# In particular, you should be able to (this is weird) split the remaining keys in half, make a multiindex for each half, and use this to build a matrix of "impact", visualizable with matshow. Splitting keys in half is not so weird as it sounds; it need not be exactly half, and we can partition each control as a "x" or "y" key manually. Depending on which parameter we're interested in, the dimensions of the matrix will change from the ideal square-ish.

# Oh of course the full matrix arrangement makes for a perfectly good matshow of the minf themselves. Definitely worth doing.

# =#

# import CSV
# import DataFrames


# infile = "surveys/surveys/rosenbrock/zeropoint/L20_M20000_σ0.0.csv"
# # df = CSV.read(infile, DataFrames.DataFrame)


# #= TODO: Build a multiindex:

# For all controls (and inputs too I guess but I imagine this is only meant for a file where inputs are all the same, so let's just say only controls for now),
#     identify all the unique values in each control. Sort them. That defines an index.
# Probably have already averaged over seeds I think maybe.

# Now for each output, build a matrix

# =#


# CONTROLS = [    # TODO: This can go in `SurveyTools`
#     :order, :n,
#     :η0, :α, :ApK,
#     :h0, :γ,
#     :NpL,
#     :trust, :tolerance,
# ]

# function multiindex(fields, df)
#     uniques = NamedTuple(field => sort(unique(df[!,field])) for field in fields)
#     CAPS = [length(unique) for unique in uniques]

#     mix = DataFrames.DataFrame()
#     for field in fields
#         mix[!,field] = indexin(df[!,field], uniques[field])
#     end

#     return mix, CAPS
# end

# """ Split controls roughly in half.
# This function gives the index of CAPS after which
#     the cumulative product is at least the square root of the total product.

# Duh.

# (I'm pretty sure you could use a cumulative sum and half the total sum,
#     but I'd need to check the algebra...)
# """
# function partition(CAPS)
#     return findfirst(cumprod(CAPS) .≥ sqrt(prod(CAPS)))
# end

# function as_index(multiindex, CAPS)
#     ix = 1
#     place = 1
#     for i in eachindex(multiindex)
#         ix += place * (multiindex[i] - 1)
#         place *= CAPS[i]
#     end
#     return ix
# end

# function as_cartesianindex(index, CAPS, part)
#     return 1 .+ reverse(divrem(index-1, prod(CAPS[1:part-1])))
# end

# function get_ticks(labels, CAPS, part)
#     ylocs = cumprod(CAPS[1:part-1])
#     xlocs = cumprod(CAPS[part:end])
#     return (
#         yticks = (ylocs, labels[1:part-1]),
#         xticks = (xlocs, labels[part:end]),
#     )
# end

# mix, CAPS = multiindex(CONTROLS, df)
# part = partition(CAPS)
# minfmatrix = ones(prod(CAPS[1:part-1]), prod(CAPS[part:end]))
# for (r, row) in enumerate(eachrow(mix))
#     ix = as_index(collect(row), CAPS)
#     i, j = as_cartesianindex(ix, CAPS, part)
#     minfmatrix[i,j] = df[r,:minf]
# end

# import Plots
# ticks = get_ticks(CONTROLS, CAPS, part)
# Plots.heatmap(1 .- minfmatrix;
#     aspect_ratio=:equal, framestyle=:origin,
#     xticks=ticks.xticks, yticks=ticks.yticks,
#     zlims=[0,1], zticks=[0,.2,.4,.6,.8,1.0])
# Plots.gui()

# #= This is really cool.

# Now let's see how to collapse one column of a dataframe into its range.
# =#

# # field = :ApK
# # keys = setdiff(CONTROLS, [field])
# # gp = DataFrames.groupby(df, keys)
# # cb = DataFrames.combine(gp, :minf => (x -> maximum(x) - minimum(x)) => :rangef)

# # mix, CAPS = multiindex(keys, cb)
# # part = partition(CAPS)
# # rangefmatrix = zeros(prod(CAPS[1:part-1]), prod(CAPS[part:end]))
# # for (r, row) in enumerate(eachrow(mix))
# #     ix = as_index(collect(row), CAPS)
# #     i, j = as_cartesianindex(ix, CAPS, part)
# #     rangefmatrix[i,j] = cb[r,:rangef]
# # end

# # import Plots
# # ticks = get_ticks(keys, CAPS, part)
# # Plots.heatmap(rangefmatrix;
# #     aspect_ratio=:equal, framestyle=:origin,
# #     xticks=ticks.xticks, yticks=ticks.yticks)
# # Plots.gui()

# #= TODO:

# This is also really cool, but the range doesn't cut it.
# I think what you are actually after is simply a re-ordering of the controls
#     such that the one of interest is the least-significant indexer of the x index.
# The actual plot is otherwise simply the f matrix.
# Maybe we explicitly add vlines to separate blocks of the target attribute.

# Yes, that feels correct. The number line is the default plot for any given CSV,
#     and you can specify a particular field of interest to generate the matrix.

# Obv. you need to figure out how to specify the range...

# =#



