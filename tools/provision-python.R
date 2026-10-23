# Have reticulate fetch uv and Python once, before any page runs.
#
# Pages declare their Python packages with reticulate::py_require(), and
# reticulate then builds the environment through uv, downloading uv itself on
# first use. On a fresh CI runner that first use lands in whichever page
# executes a Python chunk first. When that page has already loaded JuliaCall,
# the download fails ("SSL peer certificate or SSH remote key was not OK" from
# astral.sh) and the whole render stops with "Python bindings not loaded".
# Doing the download here, in a clean R process with no Julia loaded, makes
# the result the same whichever pages the freeze cache lets run.
#
# Run as a Quarto `pre-render` script. A second run finds uv cached and returns
# at once; a failure here stops the render, which is the point.
reticulate::py_require("numpy")
config <- reticulate::py_config()
cat("Python for the render:", config$python, "\n")
