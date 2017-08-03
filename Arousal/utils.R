################################################################################
# LIBRARY OF ALL USED FUNCTIONS
# Keep in mind that while some functions are compatible between datasets, others 
# have been designed to work only with CK+ dataset, such as "GetModelSets", which
# takes in account the special treatment derived from the subject-based structure
# of the dataset
################################################################################

require(mxnet)
require(matrixStats)
#require(EBImage)
library(gtools)
library(ROCR)


method.name <- "arousal"
dataset.name <- "CK+"	# CK+, FER2013, JAFFE...
dataset.type <- "augmented" # original or augmented
num.classes <- 8
train.prop <- 0.8
test.prop <- 0.1
eval.prop <- 0.1
arousal.high <- c(1, 3, 4, 5, 7)    # Anger, Disgust, Fear, Happy, Surprise
arousal.low <- c(0, 2, 6)           # Neutral, Contemtp, Sadness
image.size <- 96
iterator.batch.size <- 64     # vary to control memory usage
classifier.batch.size <- 8
classifier.iterations <- 150
method.iterations <- 100
project.root.dir <- "/media/uziel/DATA/TFG/Arousal/"
image.dir <- "/media/uziel/DATA/TFG/Data/CK+/CroppedImages/"

CNNClassifier <- function(train.iterator, eval.iterator, method.iteration){
  # Set up the model
  source(paste(project.root.dir, "symbol_resnet-28-small.R", sep = ""))
  
  # get model arquitecture
  net <- GetSymbol(1)
  
  # Set seed for reproducibility
  mx.set.seed(100)
  
  #USE GPU
  devices <- mx.gpu(0)
  
  # prefix for model storage
  prefix <- sprintf('%s_%s', dataset.name, method.name)
  # logger to store metrics during training
  logger <- mx.metric.logger$new()
  
  # WD for storing the models
  setwd(paste(project.root.dir, "Models/", sep = ""))
  
  # Train the model
  model <- mx.model.FeedForward.create(
    X                  = train.iterator,
    eval.data          = eval.iterator,
    ctx                = devices,
    symbol             = net,
    eval.metric        = mx.metric.rmse,
    num.round          = classifier.iterations,
    learning.rate      = 0.01,
    momentum           = 0.9,
    wd                 = 0.00001,
    array.batch.size   = classifier.batch.size,
    batch.end.callback = mx.callback.log.train.metric(5, logger),
    epoch.end.callback = mx.callback.save.checkpoint(prefix),
    initializer        = mx.init.Xavier(factor_type = "in", magnitude = 2.34),
    optimizer          = "sgd"
  )
  
  # metrics proportion
  metrics.proportion <- length(logger$train) / classifier.iterations
  
  # get best performing model iteration
  best.model.iteration <- round(
    tail(which(logger$eval == min(logger$eval)), n = 1) / metrics.proportion)
  
  # load best performing model iteration
  best.model <- mx.model.load(prefix, best.model.iteration)

  # save model for further comparison
  prefix <- sprintf("%s_%s_result_%s", dataset.name, method.name, method.iteration)
  mx.model.save(best.model, prefix, method.iteration)
  
  # return best model and log metrics
  return(list("model" = best.model,
              "train.error" = logger$train,
              "eval.error" = logger$eval))
}

GetPrediction <- function(model, test.iterator){
  # Predict labels
  return(predict(model, test.iterator, mx.gpu(0)))
}

GetConfMatrix <- function(predicted, test.label, threshold = 0.5){
  conf.matrix <- t(as.matrix(
    table(factor(predicted > threshold, levels=c(F, T)), test.label)))
  return(conf.matrix[c(2,1),c(2,1)])
}

GetFAR <- function(predicted, test.label, threshold = 0.5){
  # computes False Acceptance Rate
  ret <- NULL
  for(i in 1:length(threshold)){
    conf.matrix <- GetConfMatrix(predicted, test.label, threshold[i])
    FP <- conf.matrix[2]
    AN <- sum(conf.matrix[2,])
    ret <- c(ret, FP/AN)
  }
  return(ret)
}

GetFRR <- function(predicted, test.label, threshold = 0.5){
  # computes False Rejection Rate
  ret <- NULL
  for(i in 1:length(threshold)){
    conf.matrix <- GetConfMatrix(predicted, test.label, threshold[i])
    FN <- conf.matrix[3]
    AP <- sum(conf.matrix[1,])
    ret <- c(ret, FN/AP)
  }
  return(ret)
}

ComputeErrorMinimisation <- function(predicted, test.label){
  # computes the intersection that minimises the error rate
  threshold <- uniroot(function(x) 
      GetFAR(predicted, test.label, x) -
        GetFRR(predicted, test.label, x), c(0, 1))$root
  
  EER <- GetFRR(predicted, test.label, threshold)
  return(list("EER" = EER, "th" = threshold))
}

GetIterator <- function(data, name){
  # gets train iterator
  # write data records to a list file
  
  setwd(image.dir)
  
  write.table(  data,
                file = sprintf("%s_%s.lst", name, dataset.type), 
                sep = "\t",
                row.names = FALSE, 
                col.names = FALSE,
                quote = FALSE)
  
  # generates .rec with data records
  system(sprintf('~/mxnet/bin/im2rec %s_%s.lst %s %s_%s.rec color=0 resize=%d',
                 name, dataset.type, image.dir, name, dataset.type, image.size))
  
  # makes sure mean.bin doesn't exist
  if(file.exists("mean.bin")){
    file.remove("mean.bin")
  }
  
  # creates iterator
  data.iterator <- mx.io.ImageRecordIter(
      path.imgrec             = sprintf("%s_%s.rec", name, dataset.type),
      path.imglist            = sprintf("%s_%s.lst", name, dataset.type),
      batch.size              = iterator.batch.size,
      data.shape              = c(image.size, image.size, 1),
      mean.img                = "mean.bin"
    )
  
  return(data.iterator)
}

ComputeDistributedSet <- function(label, label.limit, result.set, data.for.balance){
  # sets result.set to have "label.limit" or close of "label" images
  repeat{
    # get indices of images of requested label
    indices <- which(data.for.balance[, 2] == label)
    # select a random image index
    index <- sample(indices, 1)
    # select subject from the selected image
    subject <- dirname(dirname(data.for.balance[index, 3]))
    # get indices of all images of selected subject
    indices <- with(data.for.balance,
                    grepl(paste(subject, "/*", sep = ""), data.for.balance[, 3]))
    # add all images of given subject to current set
    result.set <- rbind(result.set, data.for.balance[indices, ])
    # remove all images of given subject from original set
    data.for.balance <- data.for.balance[-(which(indices == TRUE)), ]
    
    if(sum(result.set[, 2] == label) >= label.limit){
      return(list("t0" = result.set,
                  "t1" = data.for.balance))
    }
  }
  
}

BalanceSet <- function(data){
  # set every image stack to the minimum size
  min.size <- min(sapply(0:(num.classes - 1), function(i) sum(data[, 2] == i)))
  
  for(label in 0:(num.classes - 1)){
    if(sum(data[, 2] == label) != min.size){
      # get indices of images with label
      indices <- which(data[, 2] == label)
      # sample (difference in size) indices
      samples <- sample(indices, sum(data[, 2] == label) - min.size)
      # remove sampled indices
      data <- data[-(samples), ]
    }
  }
  return(data)
}

ArousalBalance <- function(data){
  # balances data according to arousal needs. Label 1 vs Label 0
  data.arousal <- Arousal(data)
  
  if(sum(data.arousal[, 2] == 1) > sum(data.arousal[, 2] == 0)){
    label.balanced <- 0
    labels.set <- arousal.high
  } else {
    label.balanced <- 1
    labels.set <- arousal.low
  }
  
  amount <- round(sum(data.arousal[, 2] == label.balanced) / length(labels.set))
  
  images.indices <- as.vector(sapply(labels.set,
                                     function(label) sample(which(data[, 2] == label),
                                                            amount)))
  
  # get indices of images with label 1
  indices <- which(data.arousal[, 2] == label.balanced)
  result <- rbind(data.arousal[indices, ], data.arousal[images.indices, ])
  
  return(result)
}

GetDataSets <- function(data.original){
  # get test, eval and train sets with independent subjects
  # get each label amount
  label.amounts <- sapply(0:(num.classes - 1), function(i) sum(data.original[, 2] == i))
  
  check <- TRUE
  
  while(check){
    data.for.balance <- data.original
    
    test <- data.frame()
    
    for(label in 0:(num.classes - 1)){
      listed.result <- ComputeDistributedSet(label,
                                             floor(label.amounts[label + 1] * test.prop),
                                             test, data.for.balance)
      test <- listed.result$t0
      data.for.balance <- listed.result$t1
    }
    
    eval <- data.frame()
    
    for(label in 0:(num.classes - 1)){
      listed.result <- ComputeDistributedSet(label,
                                             floor(label.amounts[label + 1] * eval.prop),
                                             eval, data.for.balance)
      eval <- listed.result$t0
      data.for.balance <- listed.result$t1
    }
    
    train <- data.for.balance
    
    # check if one of the labels has 0 samples
    check <- (min(sapply(0:(num.classes - 1),
                         function(i) sum(test[, 2] == i))) == 0 |
                min(sapply(0:(num.classes - 1),
                           function(i) sum(eval[, 2] == i))) == 0 |
                min(sapply(0:(num.classes - 1),
                           function(i) sum(train[, 2] == i))) == 0)
  }
  
  # balance set
  test <- BalanceSet(test)
  test <- ArousalBalance(test)
  # shuffle data
  test <- test[sample(1:nrow(test)),]
  # compute iterator
  test.label <- test[, 2]
  test.iterator <- GetIterator(test, "test")
  
  # balance set
  eval <- BalanceSet(eval)
  eval <- ArousalBalance(eval)
  # shuffle data
  eval <- eval[sample(1:nrow(eval)),]
  # compute iterator
  eval.iterator <- GetIterator(eval, "eval")
  
  # balance set
  train <- BalanceSet(train)
  train <- ArousalBalance(train)
  # shuffle data
  train <- train[sample(1:nrow(train)),]
  # compute iterator
  train.iterator <- GetIterator(train, "train")
  
  return(list("train.iterator" = train.iterator,
              "eval.iterator" = eval.iterator,
              "test.iterator" = test.iterator,
              "test.label" = test.label))
}

GetModelSets <- function(data.original){
  # computes train and tests sets for training model
  data.original <- data.original[sample(1:nrow(data.original)),]
  
  # get model sets
  model.sets <- GetDataSets(data.original)
  return(model.sets)
}

GetMetrics <- function(predicted, test.label, threshold){
  # computes all relevant evaluation metrics
  # get confusion matrix
  conf.matrix <- GetConfMatrix(predicted, test.label, threshold)
  # computes confusion matrix metrics
  TP <- conf.matrix[1]
  FP <- conf.matrix[2]
  FN <- conf.matrix[3]
  TN <- conf.matrix[4]
  N <- sum(conf.matrix)
  AP <- sum(conf.matrix[1,])
  AN <- sum(conf.matrix[2,])
  PD <- sum(conf.matrix[,1])
  ND <- sum(conf.matrix[,2])
  # computes final metrics
  # accuracy
  ACC <- (TP + TN) / N
  # sensitivity
  SEN <- TP / AP
  # specificity
  SPE <- TN / AN
  # positive predictive value
  PPV <- TP / PD
  # negative predictive value
  NPV <- TN / ND
  # false positive rate
  FPR <- FP / AN
  # false discovery rate
  FDR <- FP / (TP + FP)
  # F1 score
  F1S <- 2 * ((PPV * SEN) / (PPV + SEN))
  
  return(c(ACC, SEN, SPE, PPV, NPV, FPR, FDR, F1S))
}

ComputeEERCurves <- function(predicted, test.label){
  FAR.curve <- sapply(seq(0, 1, by = 0.001),
                      function(threshold)
                        GetFAR(predicted, test.label, threshold))
  
  FRR.curve <- sapply(seq(0, 1, by = 0.001),
                      function(threshold)
                        GetFRR(predicted, test.label, threshold))
  return(list("FAR" = FAR.curve, "FRR" = FRR.curve))
}

ComputeROCCurve <- function(predictions, labels){
  pred <- prediction(predictions, labels)
  ROC.curve <- performance(pred, "tpr", "fpr")
  return(ROC.curve)
}

ComputeLabelIteration <- function(data.original, method.iteration){
  
  # compute train and test sets
  model.sets <- GetModelSets(data.original)
  
  # run classifier with train set
  training.result <- CNNClassifier(model.sets$train.iterator,
                                   model.sets$eval.iterator,
                                   method.iteration)
  
  # get model
  model <- training.result$model
  
  # get predicted values
  test.label <- model.sets$test.label

  ###############################
  # begin bug workaround (predicted size is not coherent)
  repeat{
    predicted <- GetPrediction(model, model.sets$test.iterator)
    Sys.sleep(0.01)
    if(length(predicted) > length(test.label) * 0.98){
      break
    }
  }
  
  if(length(predicted) < length(test.label)){
    size.left <- length(test.label) - length(predicted)
    test.label <- test.label[-(1:size.left)]
  }
  # end bug workaround
  ###############################

    # get EER & Threshold
  error.minimisation <- ComputeErrorMinimisation(predicted, test.label)
  
  # get FRR & FAR curves
  EER.curves <- ComputeEERCurves(predicted, test.label)

  # get ACC, SEV, SP, ... in form of a list
  metrics <- GetMetrics(predicted, test.label, error.minimisation$th)
  # return all metrics
  return(list("metrics" = c(error.minimisation$EER,
                            error.minimisation$th, metrics),
            "train.error" = training.result$train.error,
            "eval.error" = training.result$eval.error,
            "FAR" = EER.curves$FAR,
            "FRR" = EER.curves$FRR,
            "predicted" = predicted,
            "label" = test.label))
}

ReorderVector <- function(v1, v2){
  # alternate columns from v1 and v2
  v3 <- NULL
  for (i in 1:length(v1)) v3 <- c(v3, v1[i], v2[i])
  return(v3)
}

GetSmallestLength <- function(list.to.reduce){
  list.lengths <- sapply(1:length(list.to.reduce),
                         function(x) length(unlist(list.to.reduce[x])))
  return(min(list.lengths))
}

CutVector <- function(vector, size){
  length(vector) <- size
  return(vector)
}

CutVectorList <- function(list.to.cut, size){
  vector.cut <- t(sapply(1:length(list.to.cut),
                         function(x) CutVector(unlist(list.to.cut[x]), size)))
  return(vector.cut)
}

UnlistErrors <- function(errors.list){
  # get method errors
  # get model train error
  smallest.length <- GetSmallestLength(errors.list[, 1])
  train.error <- CutVectorList(errors.list[, 1], smallest.length)
  
  # get model eval error
  smallest.length <- GetSmallestLength(errors.list[, 2])
  eval.error <- CutVectorList(errors.list[, 2], smallest.length)
  
  return(list("train.error" = train.error, "eval.error" = eval.error))
}

StoreBestModel <- function(eval.error){
  # obtain iteration from the best performing model
  best.model.method.iteration <- tail(
    which(eval.error == max(eval.error), arr.ind = TRUE), n = 1)[1]
  # load best model
  setwd(paste(project.root.dir, "Models/", sep = ""))
  prefix <- sprintf("%s_%s_result_%s", dataset.name, method.name, best.model.method.iteration)
  best.model.final <- mx.model.load(prefix, best.model.method.iteration)
  prefix <- sprintf("%s_%s_final_%s",
                    dataset.name, method.name, dataset.type)
  mx.model.save(best.model.final, prefix, best.model.method.iteration)
  
}

StoreValidationGraph <- function(train.error, eval.error){
  # average results
  train.error <- colMeans(train.error)
  eval.error <- colMeans(eval.error)
  
  # save validation curve plot
  plot.dir <- paste(project.root.dir, "Plots/", sep = "")
  plot.filename <- paste(plot.dir,
                         sprintf("%s_%s_%s.png",
                                 dataset.name, method.name, dataset.type),
                         sep = "")
  png(filename = plot.filename)
  plot(c(0, classifier.iterations), c(0, 1), type = "n",
       xlab = "Epochs", ylab = "Error (RMSE)",
       main = sprintf("(%s) - Validation results",
                      dataset.name))
  x.axis <- seq(0, classifier.iterations, by = classifier.iterations / length(train.error))[-1]
  lines(x.axis, train.error, col = "red", lwd = 2.5)
  eval.error <- c(rep(NA, each = (length(train.error) - length(eval.error))), eval.error)
  lines(x.axis, eval.error, col = "blue", lwd = 2.5)
  legend(classifier.iterations / 2, 0.8,
         c("Training Error", "Evaluation Error"),
         lty = c(1, 1), lwd = c(2.5, 2.5), col = c("red", "blue"))
  dev.off()
}

UnlistEER <- function(EER.list){
  # get method errors
  # get FAR curves
  FAR <- matrix(unlist(EER.list[, 1]),
                        ncol = length(unlist(EER.list[1, 1])),
                        byrow = TRUE)
  
  # get FRR curves
  FRR <- matrix(unlist(EER.list[, 2]),
                       ncol = length(unlist(EER.list[1, 2])),
                       byrow = TRUE)
  
  return(list("FAR" = FAR, "FRR" = FRR))
}

StoreEERGraph <- function(FAR, FRR){
  # average results
  FAR <- colMeans(FAR)
  FRR <- colMeans(FRR)
  
  # save validation curve plot
  plot.dir <- paste(project.root.dir, "Plots/", sep = "")
  plot.filename <- paste(plot.dir,
                         sprintf("%s_%s_%s_EER.png",
                                 dataset.name, method.name, dataset.type),
                         sep = "")
  png(filename = plot.filename)
  plot(c(0, 1), c(0, 1), type = "n",
       ylab = "Error", xlab = "Threshold",
       main = sprintf("(%s) - Equal Error Rate & Optimum Threshold",
                      dataset.name))
  lines(seq(0, 1, by = 0.001), FAR, col = "green", lwd = 2.5)
  lines(seq(0, 1, by = 0.001), FRR, col = "red", lwd = 2.5)
  legend(0.5, 0.6,
         c("False Acceptance Rate", "False Rejection Rate"),
         lty = c(1, 1), lwd = c(2.5, 2.5), col = c("green", "red"))
  dev.off()
}

StoreROCGraph <- function(graph){
  # save validation curve plot
  plot.dir <- paste(project.root.dir, "Plots/", sep = "")
  plot.filename <- paste(plot.dir,
                         sprintf("%s_%s_%s_ROC.png",
                                 dataset.name, method.name, dataset.type),
                         sep = "")
  png(filename = plot.filename)
  plot(0,
       xlim = c(0, 1), ylim = c(0, 1), type = "n",
       xlab = "False Positive Rate", ylab = "True Positive Rate",
       main = sprintf("(%s) - Receiver Operating Characteristic (ROC)",
                      dataset.name))
  # plot ROC
  plot(graph, avg = "vertical", col = "blue", lwd = 2.5)
  # plot diagonal line
  lines(x = c(0, 1), y = c(0, 1), lwd = 3)
  legend(0.6, 0.1, c("Emotion Activation"), lty = 1, lwd = 2.5, col = "blue")
  dev.off()
}

GetLabelMetrics <- function(data.original){
  # get metrics for one label, average of multiple iterations
  # get label metrics over iterations
  method.result <- t(sapply(1:method.iterations, function(method.iteration)
    ComputeLabelIteration(data.original, method.iteration)))
  
  # get errors
  unlisted.errors <- UnlistErrors(method.result[, 2:3])
  
  # get and store best model of all method.iterations
  StoreBestModel(unlisted.errors$eval.error)

  # compute and store validation results graph
  StoreValidationGraph(unlisted.errors$train.error, unlisted.errors$eval.error)
    
  # get EER curves
  unlisted.EER <- UnlistEER(method.result[, 4:5])
  
  # compute and store EER graph
  StoreEERGraph(unlisted.EER$FAR, unlisted.EER$FRR)
  
  # get ROC curves
  ROC.curve <- ComputeROCCurve(method.result[, 6], method.result[, 7])
  
  # compute and store ROC graph
  StoreROCGraph(ROC.curve)
  
  # get method metrics
  method.metrics <- matrix(unlist(method.result[, 1]),
                           ncol = length(unlist(method.result[1, 1])),
                           byrow = TRUE)

  # return mean and variance of all iterations
  # METRIC1, VARIANCE2, METRIC2, VARIANCE2, ...
  return(ReorderVector(colMeans(method.metrics), colVars(method.metrics)))
}

Arousal <- function(data.original){
  # arousal --> high activation vs low activation
  # get high ativation indices
  high.indices <- data.original[, 2] %in% arousal.high
  
  # get low activation indices
  low.indices <- data.original[, 2] %in% arousal.low
    
  data.original[high.indices, 2] <- 1
  data.original[low.indices, 2] <- 0
  return(data.original)
}

GetMethodMetrics <- function(data.original){
  # get metrics for each label
  return(GetLabelMetrics(data.original))
}

MetricsStorage <- function(metrics){
  # create metrics dataframe
  metrics.data.frame <- data.frame(t(metrics))
  # assign column labels to data.frame
  names(metrics.data.frame) <- c("EER", "Var(EER)",
                                 "Th", "Var(Th)",
                                 "ACC", "Var(ACC)",
                                 "SEN", "Var(SEN)",
                                 "SPE", "Var(SPE)",
                                 "PPV", "Var(PPV)",
                                 "NPV", "Var(NPV)",
                                 "FPR", "Var(FPR)",
                                 "FDR", "Var(FDR)",
                                 "F1S", "Var(F1S)")
  setwd(project.root.dir)
  write.csv(metrics.data.frame,
            sprintf("%s_%s_metrics_dataset_%s.csv",
                    dataset.name, method.name, dataset.type),
            row.names = FALSE)  
}

GetAllArousalMetrics <- function(data.original){
  # compute all metrics
  all.method.metrics <- GetMethodMetrics(data.original)
  
  # storage of metrics
  MetricsStorage(all.method.metrics)
}