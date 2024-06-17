include("functions.jl")
    import .TestFunctions

include("surveys.jl")
    import .CensusTemplates
    import .CensusTools

outdir = "surveys/censes/rosenbrock/zeropoint"
outfile = "$outdir/BROAD_L20_M20000_σ0.01.csv"

open(outfile, "w") do io
    CensusTools.run!(io,
        CensusTemplates.BROAD,
        TestFunctions.Rosenbrock.lossfunction(),
        zeros(20);
        M=20000,
        σ = 0.01,
        seed = 0,
    )
end