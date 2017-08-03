################################################################################
# Arousal
# Trains CNN model in arousal mode. Trains one CNN for high and low activation
# emotions. Writes metrics as a result of the training phase.
################################################################################

require(tictoc)

# Clean workspace
rm(list=ls())

# set root dir
root.dir = "/media/uziel/DATA/TFG/"

# source utils library
source(paste(root.dir, "Arousal/utils.R", sep = ""))

# load original set of images
data.original <- read.csv(paste(root.dir, "Data/CK+/ck_extended_record.csv", sep = ""),
                          stringsAsFactors = FALSE,
                          header = FALSE)

tic()
all.metrics <- GetAllArousalMetrics(data.original)
toc()