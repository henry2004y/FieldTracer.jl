using FieldTracer
using Documenter

makedocs(;
    modules=[FieldTracer],
    authors="Hongyang Zhou <hyzhou@umich.edu>",
    sitename="FieldTracer.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://henry2004y.github.io/FieldTracer.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Example" => "example.md",
        "Log" => "log.md",
        "Internal" => "internal.md",
    ],
)

deploydocs(;
    repo="github.com/henry2004y/FieldTracer.jl",
)
