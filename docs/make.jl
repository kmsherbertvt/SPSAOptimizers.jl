using SPSAOptimizers
using Documenter

DocMeta.setdocmeta!(SPSAOptimizers, :DocTestSetup, :(using SPSAOptimizers); recursive=true)

makedocs(;
    modules=[SPSAOptimizers],
    authors="Kyle Sherbert <kyle.sherbert@vt.edu> and contributors",
    sitename="SPSAOptimizers.jl",
    format=Documenter.HTML(;
        canonical="https://kmsherbertvt.github.io/SPSAOptimizers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kmsherbertvt/SPSAOptimizers.jl",
    devbranch="main",
)
