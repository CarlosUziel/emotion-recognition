################################################################################
# Verification
# Trains CNN model in verification mode. Trains one CNN for each class.
# Writes metrics as a result of the training phase.
################################################################################

require(tictoc)

# Clean workspace
rm(list=ls())

# set root dir
root.dir = "/media/uziel/DATA/TFG/"

# source utils library
source(paste(root.dir, "Verification/utils.R", sep = ""))

# load original set of images
data.original <- read.csv(paste(root.dir, "Data/CK+/ck_extended_record.csv", sep = ""),
                          stringsAsFactors = FALSE,
                          header = FALSE)

# remove comtemtp
#data.original <- data.original[-(which(data.original[, 2] == 2)), ]

# adjust the rest of the labels
#for(i in 3:7){
#  data.original[which(data.original[, 2] == i), 2] <- i - 1
#}

tic()
all.metrics <- GetAllVerificationMetrics(data.original)
toc()