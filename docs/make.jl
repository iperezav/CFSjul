using Documenter
using ChenFliessSeries

makedocs(
    sitename = "ChenFliessSeries Documentation",
    pages = [
        "Index" => "index.md",
        "An other page" => "anotherPage.md",
    ],
    format = Documenter.HTML(),
    modules = [ChenFliessSeries]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/iperezav/ChenFliessSeries.jl.git",
    devbranch = "main"
)
