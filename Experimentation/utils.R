image.size <- 96
num.classes <- 8
iterator.batch.size <- 8

crop.script.path = "/media/uziel/DATA/TFG/FaceRecognition/crop_face.py"
root.dir = "/media/uziel/DATA/TFG/ExperimentResults/"
source.dir <- paste(root.dir, "Images/", sep = "")
destination.dir <- paste(root.dir, "CroppedImages/", sep = "")
classification.models <-"/media/uziel/DATA/TFG/BestModels/Classification/"
verification.models <- "/media/uziel/DATA/TFG/BestModels/Verification/"
arousal.models <- "/media/uziel/DATA/TFG/BestModels/Arousal/"
pred.dir <- "/media/uziel/DATA/TFG/ExperimentResults/Predictions/"
groups.dir <- "/media/uziel/DATA/TFG/ExperimentResults/DataGroups/"
corr.dir <- "/media/uziel/DATA/TFG/ExperimentResults/Correlations/"

ExtractFrames <- function(video.file, image.dir, video.code){
  # create video image dir
  system(sprintf("mkdir %s", image.dir))
  
  # https://trac.ffmpeg.org/wiki/Create%20a%20thumbnail%20image%20every%20X%20seconds%20of%20the%20video
  # Output one image every 10 second, named video%s_image1.png, video%s_image.png, video%s_image.png, etc.
  system(sprintf("ffmpeg -i %s -vf fps=10 %s/video_%d_image_%%d.png",
         video.file, image.dir, video.code))
}

GetImagesRecord <- function(image.dir){
  # list all images
  image.files <- mixedsort(list.files(image.dir, pattern = "*.png"))
  # index
  index <- array(1:length(image.files))
  # labels
  labels <- array(1:length(image.files))
  # compose records
  records <- data.frame(index, labels, image.files, stringsAsFactors = FALSE)
  return(records)
}

GetIterator <- function(image.dir, images.record, video.code){
  # gets images iterator
  # write data records to a list file
  file <- paste(image.dir, sprintf("experiment_%s", video.code), sep = "")
  write.table(  images.record,
                file = paste(file, ".lst", sep = ""), 
                sep = "\t", 
                row.names = FALSE, 
                col.names = FALSE,
                quote = FALSE)
  
  # set working directory
  setwd(image.dir)
  
  # generates .rec with data records
  system(sprintf('~/mxnet/bin/im2rec %s.lst %s %s.rec color=0 resize=%d',
                 file, image.dir, file, image.size))
  
  # creates iterator
  data.iterator <- mx.io.ImageRecordIter(
    path.imgrec             = sprintf("%s.rec", file),
    path.imglist            = sprintf("%s.lst", file),
    batch.size              = iterator.batch.size,
    data.shape              = c(image.size, image.size, 1),
    mean.img                = "mean.bin"
  )
  
  return(data.iterator)
}

GetPrediction <- function(model, test.iterator, n.images){
  # Predict labels
  predicted <- predict(model, test.iterator, mx.gpu(0))
  return(predicted)
}

LoadModel <- function(model.dir, pattern){
  # load model
  model <- list.files(model.dir, pattern = pattern)
  name.split <- unlist(strsplit(model[1], split = "-"))
  prefix <- name.split[1]
  iteration <- as.numeric(file_path_sans_ext(name.split[2]))
  setwd(model.dir)
  model <- mx.model.load(prefix, iteration)
  return(model)
}

GetClassificationPrediction <- function(images.iterator, n.images){
  # load model
  best.model <- LoadModel(classification.models, "")
  
  # predict
  ###############################
  # begin bug workaround (predicted size is not coherent)
  repeat{
    predicted <- GetPrediction(best.model, images.iterator, n.images)
    Sys.sleep(0.01)
    if(ncol(predicted) == n.images){
      break
    }
  }
  # end bug workaround
  ###############################
  
  predicted <- data.frame(t(predicted))

  # return predictions
  return(predicted)
}

VerificationLabel <- function(images.iterator, label, n.images){
  # load model
  best.model <- LoadModel(verification.models, sprintf("label_%s", label))

  # predict
  ###############################
  # begin bug workaround (predicted size is not coherent)
  repeat{
    predicted <- GetPrediction(best.model, images.iterator, n.images)
    Sys.sleep(0.01)
    if(ncol(predicted) == n.images){
      break
    }
  }
  # end bug workaround
  ###############################
  # return predictions
  return(predicted)
}

GetVerificationPrediction <- function(images.iterator, n.images){
  predicted <- sapply(0:(num.classes - 1),
                      function(label) VerificationLabel(images.iterator, label, n.images))
  predicted <- data.frame(predicted)
  return(predicted)
}

GetArusalPrediction <- function(images.iterator, n.images){
  # load model
  best.model <- LoadModel(arousal.models, "")
  
  # predict
  ###############################
  # begin bug workaround (predicted size is not coherent)
  repeat{
    predicted <- GetPrediction(best.model, images.iterator, n.images)
    Sys.sleep(0.01)
    if(ncol(predicted) == n.images){
      break
    }
  }
  # end bug workaround
  ###############################

  predicted <- data.frame(t(predicted))

  # return predictions
  return(predicted)
}

StorePredictions <- function(predictions, image.dir, video.code, method.name){
  
  if(method.name == "arousal"){
    colnames(predictions) <- "Arousal"
  }else{
    colnames(predictions) <- paste("Label", c(0:(num.classes - 1)))
  }
  
  write.table(predictions,
              file = paste(pred.dir,
                           sprintf("%s/video_%s_%s_predictions.csv",
                                   video.code, video.code, method.name),
                           sep = ""),
              sep = ",",
              quote = FALSE,
              row.names = FALSE)
}

ComputePredictions <- function(image.dir, images.iterator, video.code, n.images){
  system(sprintf("mkdir %s%s/", pred.dir, video.code))
  
  # compute classficiation predictions
  classification.prediction <- GetClassificationPrediction(images.iterator, n.images)
  StorePredictions(classification.prediction, image.dir, video.code, "classification")

  # compute verification predictions
  verification.prediction <- GetVerificationPrediction(images.iterator, n.images)
  StorePredictions(verification.prediction, image.dir, video.code, "verification")
  
  # compute arousal predictions
  arousal.prediction <- GetArusalPrediction(images.iterator, n.images)
  StorePredictions(arousal.prediction, image.dir, video.code, "arousal")
}

PredictFrames <- function(image.dir, video.code){
  # get images record
  images.record <- GetImagesRecord(image.dir)
  # get images iterator
  images.iterator <- GetIterator(image.dir, images.record, video.code)
  # obtain and write predictions
  ComputePredictions(image.dir, images.iterator, video.code, nrow(images.record))
}

CropFace <- function(image.file, i){
  # set origin path
  origin.path <- paste(source.dir, image.file, sep = "")
  # set store path
  store.dir <- paste(destination.dir, dirname(image.file), sep = "")
  store.path <- paste(store.dir, "/", basename(image.file), sep = "")
  
  # create video directory
  system(sprintf("mkdir %s -p", store.dir))
  
  # crop image
  system(sprintf("python %s %s %s", crop.script.path, origin.path, store.path))
  
  print(sprintf("Done cropping image: %d", i))
}

ReorderVector <- function(v1, v2){
  # alternate columns from v1 and v2
  v3 <- NULL
  for (i in 1:length(v1)) v3 <- c(v3, v1[i], v2[i])
  return(v3)
}

GetSubjectScores <- function(score.file, ad.interval){
  # load file
  scores <- read.csv(paste(pred.dir, score.file, sep = ""),
                     stringsAsFactors = FALSE,
                     header = TRUE)
  
  # if interval exceeds available data, return NA
  if(ad.interval[2] > nrow(scores)){
    return(rep(NA, ncol(scores)))
  }
  
  # return average of rows between the interval
  scores <- data.matrix(scores[ad.interval[1]:ad.interval[2],])
  return(ReorderVector(colMeans(scores), colVars(scores)))
}

GroupMethodData <- function(ad.name, ad.interval, method.name){
  # get all method.name predictions
  scores.files <- mixedsort(list.files(pred.dir,
                                       pattern = method.name,
                                       recursive = TRUE))

  # for each subject, compute average scores
  scores <- sapply(1:length(scores.files),
                   function(i) GetSubjectScores(scores.files[i], ad.interval))
  
  scores <- t(data.frame(scores))

  if(method.name == "arousal"){
    colnames(scores) <- c("Arousal", "Var(Arousal)")
  } else{
    labels <- paste("Label.", c(0:(num.classes - 1)), sep = "")
    labels.vars <- paste("Var(Label.", c(0:(num.classes - 1)), ")", sep = "")
    colnames(scores) <- ReorderVector(labels, labels.vars)
  }
  write.csv(scores,
            sprintf("%s%s_%s.csv",
                    groups.dir, ad.name, method.name),
            row.names = FALSE)
}

GroupData <- function(ad.name, ad.interval){
  method.names <- c("classification",
                    "verification",
                    "arousal")
  
  for(i in 1:length(method.names)){
    GroupMethodData(ad.name, ad.interval, method.names[i])
  }
}

ComputeAdCorrelation <- function(ad, variable){
  # read ad data
  ad.data <- read.csv(ad,
                      stringsAsFactors = FALSE,
                      header = TRUE)
  
  # read variable data
  var.data <- read.csv(variable,
                       stringsAsFactors = FALSE,
                       header = FALSE)
  
  # remove conflictive subjects
  ad.data <- ad.data[-15,]
  var.data <- var.data[-15,]
  
  
  corr <- sapply(seq(1, ncol(ad.data), 2), function(i) cor(ad.data[, i], var.data))
  return(corr)
}

ComputeVariableCorrelations <- function(variable, method.name){
  # select ads related to the variable
  ads <- mixedsort(list.files(paste(root.dir, "DataGroups/", sep = ""),
                              pattern = method.name,
                              full.names = TRUE))
  # extract number from variable
  var.name <- file_path_sans_ext(basename(variable))
  ad.number <- gsub("[^0-9]", "", var.name)
  
  if(ad.number != ""){
    ads <- subset(ads, grepl(paste("Anuncio_", ad.number, sep = ""),
                             ads))
  }
  
  # compute correlations between ad data and variable data
  v.corr <- sapply(1:length(ads), function(i) ComputeAdCorrelation(ads[i], variable))
  
  v.corr <- data.frame(v.corr)

  if(method.name == "arousal"){
    colnames(v.corr) <- "Arousal"
  } else{
    v.corr <- t(v.corr)
    colnames(v.corr) <- paste("Label.", c(0:(num.classes - 1)), sep = "")
  }

  write.csv(v.corr,
            sprintf("%s%s/%s_%s.csv",
                    corr.dir, toupper(method.name),
                    var.name, method.name),
            row.names = FALSE)
}

ComputeCorrelations <- function(method.name){
  # create method.name dir
  system(sprintf("mkdir %s%s/", corr.dir, toupper(method.name)))
  
  # load all variables data
  variables <- mixedsort(list.files(paste(root.dir, "Variables/", sep = ""),
                                    full.names = TRUE))
  
  for(variable in variables){
    ComputeVariableCorrelations(variable, method.name)
  }
  
}