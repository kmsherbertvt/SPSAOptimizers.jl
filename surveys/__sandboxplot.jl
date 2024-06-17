import Plots
include("plotting/matshow.jl")
    import .MatrixPlots

import DataFrames, CSV
# infile = "surveys/surveys/rosenbrock/zeropoint/L20_M20000_σ0.0.csv"
# df = CSV.read(infile, DataFrames.DataFrame)

# xfields = [:tolerance, :n]
# yfields = [:η0, :α, :ApK, :order]

# plt = MatrixPlots.make_plot(df, :η0, :α, xfields, yfields, :fp)





infile = "surveys/censes/rosenbrock/zeropoint/BROAD_L20_M20000_σ0.01.csv"
df = CSV.read(infile, DataFrames.DataFrame)

xfields = [:η0, :h0, :NpL, :n, :trust]
yfields = [:ApK, :α, :γ, :order, :tolerance]


xfields = [:h0, :η0, :NpL, :n, :order]
yfields = [:γ, :α, :ApK, :p, :trust, :tolerance]

multiindex = MatrixPlots.make_multiindex(df, [xfields..., yfields...])
xbases = MatrixPlots.make_bases(multiindex, xfields)
ybases = MatrixPlots.make_bases(multiindex, yfields)

plt = MatrixPlots.make_plot(df, xfields, yfields, :fp)

