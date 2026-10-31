# Rebuild chapters/data/faithful.csv from R's copy of the Old Faithful data.
#
# The generative models chapter reads a local copy of `datasets::faithful`,
# 272 eruptions of the Old Faithful geyser: each eruption's length and the
# wait until the next one, in minutes. This writes the data frame unchanged,
# with its two column names as the header.
#
# Usage: Rscript tools/get-faithful.R

out <- file.path("chapters", "data", "faithful.csv")
n_rows <- nrow(datasets::faithful)
readr::write_csv(datasets::faithful, out)
message("wrote ", n_rows, " eruptions to ", out)
