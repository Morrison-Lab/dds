# Set up a page whose Python chunks use PyTorch.
#
# A page sources this file from its first R chunk, after attaching reticulate,
# then calls use_torch() with its other Python packages.
# use_torch() declares torch along with those packages,
# then applies tools/reticulate_logging_patch.py,
# without which every Python chunk after `import torch`
# fails in reticulate's output capture (that file says why).
use_torch <- function(
  packages = character(),
  patch = "../tools/reticulate_logging_patch.py"
) {
  reticulate::py_require(c(packages, "torch"))
  reticulate::py_run_file(patch, local = TRUE)
  invisible(NULL)
}
