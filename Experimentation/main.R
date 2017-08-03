# Clean workspace
rm(list=ls())

require(gtools)
require(tools)
require(matrixStats)
require(mxnet)
source("/media/uziel/DATA/TFG/Experimentation/utils.R")

# root.dir folder of all experimentation data
root.dir = "/media/uziel/DATA/TFG/ExperimentResults/"

# obtain all Videos filenames
video.files <- mixedsort(list.files(paste(root.dir, "Videos/", sep = ""),
                                    pattern = "*.wmv",
                                    full.names = TRUE))

# create Images dir
system(sprintf("rm -rf %sImages/", root.dir))
system(sprintf("mkdir %sImages/", root.dir))

# for each video, extract frames
for (i in 1:length(video.files)){
  ExtractFrames(video.files[i], paste(root.dir, "Images/", i, sep = ""), i)
}

# for all extracted frames, apply processing if needed (resizing, color channels, face cropping...)
image.files <- mixedsort(list.files(paste(root.dir, "Images/", sep = ""),
                                    pattern = "*.png",
                                    recursive = TRUE))

# create CroppedImages dir
system(sprintf("rm -rf %sCroppedImages/", root.dir))
system(sprintf("mkdir %sCroppedImages/", root.dir))

for(i in 1:length(image.files)){
  CropFace(image.files[i], i)
}

system(sprintf("rm -rf %sPredictions/", root.dir))
system(sprintf("mkdir %sPredictions/", root.dir))

# for all subject's images, predict an emotion
for (i in 1:length(video.files)){
  PredictFrames(paste(root.dir, "CroppedImages/", i, "/", sep = ""), i)
}

system(sprintf("rm -rf %sDataGroups/", root.dir))
system(sprintf("mkdir %sDataGroups/", root.dir))

# sort predictions for all advertisement videos
ads.names <- c("Anuncio_3.1", "Anuncio_4.1", "Anuncio_1.1",
               "Anuncio_2.1", "Anuncio_6.1", "Anuncio_5.1",
               "Anuncio_3.2", "Anuncio_6.2", "Anuncio_1.2",
               "Anuncio_2.2", "Anuncio_4.2", "Anuncio_5.2",
               "Anuncio_4.3", "Anuncio_1.3", "Anuncio_5.3",
               "Anuncio_3.3", "Anuncio_6.3", "Anuncio_2.3")

s0 <- seq(15, 810, by = 45)
s0 <- s0 + 1
s1 <- seq(45, 810, by = 45)

for(i in 1:length(ads.names)){
  GroupData(ads.names[i], c(s0[i], s1[i]))
}

# remove and create dir
system(sprintf("rm -rf %sCorrelations/", root.dir))
system(sprintf("mkdir %sCorrelations/", root.dir))


method.names <- c("classification",
                  "verification",
                  "arousal")

for(i in 1:length(method.names)){
  ComputeCorrelations(method.names[i])
}

#########################################################
# MISC - Predictions from Arousal Model (for plotting)
#########################################################
pred.dir <- "/media/uziel/DATA/TFG/ExperimentResults/Predictions/"

# rest and stimuli intervals for 10 fps
r0 <- seq(0, 825, by = 45)
r0 <- r0 + 1
r1 <- seq(15, 825, by = 45)
s0 <- seq(15, 810, by = 45)
s0 <- s0 + 1
s1 <- seq(45, 810, by = 45)

# read all arousal predictions
arousal.pred <- mixedsort(list.files(pred.dir,
                                     pattern = "arousal",
                                     recursive = TRUE))

# separate rest frames and stimuli frames
for(i in 1:length(arousal.pred)){
  a.pred <- read.csv(paste(pred.dir, arousal.pred[i], sep = ""),
                     stringsAsFactors = FALSE,
                     header = TRUE)[1:825,]
  # remove high outliers
  q <- as.numeric(quantile(a.pred, na.rm = TRUE))
  outer.limit <- q[4] + (q[4] - q[2]) * 3
  ind <- a.pred > outer.limit
  a.pred[ind] <- outer.limit
  
  rest <- as.vector(sapply(1:length(r0), function(i) r0[i]:r1[i]))
  result.rest <- a.pred
  result.rest[-rest] <- 0

  result.stimuli <- a.pred
  result.stimuli[rest] <- 0
  
  write.csv(data.frame(result.rest, result.stimuli),
            sprintf("%s/%s_%s.csv",
                    pred.dir, dirname(arousal.pred[i]), "stimuli"),
            row.names = FALSE)
}

# compute medians
# read all arousal predictions
arousal.pred <- mixedsort(list.files(pred.dir,
                                     pattern = "arousal",
                                     recursive = TRUE))

ArousalAux <- function(i){
  a.pred <- read.csv(paste(pred.dir, arousal.pred[i], sep = ""),
                     stringsAsFactors = FALSE,
                     header = TRUE)[1:825,]
  
  return(a.pred)
}

medians <- rowMedians(sapply(1:length(arousal.pred), function(i) ArousalAux(i))[,-c(5, 15)])

rest <- as.vector(sapply(1:length(r0), function(i) r0[i]:r1[i]))
medians.rest <- medians
medians.rest[-rest] <- 0

medians.stimuli <- medians
medians.stimuli[rest] <- 0

write.csv(data.frame(medians.rest, medians.stimuli),
          sprintf("%s/%s.csv",
                  pred.dir, "arousal_medians"),
          row.names = FALSE)
