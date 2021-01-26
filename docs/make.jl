using FieldTracer
using Documenter

makedocs(;
    modules=[FieldTracer],
    authors="Hongyang Zhou <hyzhou@umich.edu>",
    repo="https://github.com/henry2004y/FieldTracer.jl/blob/{commit}{path}#L{line}",
    sitename="FieldTracer.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://henry2004y.github.io/FieldTracer.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/henry2004y/FieldTracer.jl",
)
