################################################################################
# LIBRARY OF ALL USED FUNCTIONS
# Keep in mind that while some functions are compatible between datasets, others 
# have been designed to work only with CK+ dataset, such as "GetModelSets", which
# takes in account the special treatment derived from the subject-based structure
# of the dataset
################################################################################

require(mxnet)
require(matrixStats)
require(gtools)

method.name <- "classification"
dataset.name <- "CK+"	# CK+, FER2013, JAFFE...
dataset.type <- "augmented" # original or augmented
num.classes <- 8 # 7 if contempt removed
train.prop <- 0.8
test.prop <- 0.1
eval.prop <- 0.1
image.size <- 96
iterator.batch.size <- 64     # vary to control memory usage
classifier.batch.size <- 8
classifier.iterations <- 150
method.iterations <- 100
project.root.dir <- "/media/uziel/DATA/TFG/Classification/"
image.dir <- "/media/uziel/DATA/TFG/Data/CK+/CroppedImages/"


CNNClassifier <- function(train.iterator, eval.iterator, method.iteration){
  # Set up the model
  source(paste(project.root.dir, "symbol_resnet-28-small.R", sep = ""))
  
  # get model arquitecture
  net <- GetSymbol(num.classes)

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
    eval.metric        = mx.metric.accuracy,
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
    tail(which(logger$eval == max(logger$eval)), n = 1) / metrics.proportion)
  
  # load best performing model iteration
  best.model <- mx.model.load(prefix, best.model.iteration)

  # save model for further comparison
  prefix <- sprintf("%s_%s_result_%s", dataset.name, method.name, method.iteration)
  mx.model.save(best.model, prefix, method.iteration)
  
  # return best model and log metrics
  return(list("model" = best.model, "train.error" = logger$train, "eval.error" = logger$eval))
}

GetPrediction <- function(model, test.iterator){
  # Predict labels
  predicted <- predict(model, test.iterator, mx.gpu(0))
  # Assign labels
  predicted <- max.col(t(predicted)) - 1
  return(predicted)
}

GetConfMatrix <- function(predicted, test.label, label){
  # computes confusion matrix
  label <- label + 1
  conf.matrix <- t(as.matrix(table(factor(predicted, levels=c(0:(num.classes - 1))),
                                   factor(test.label, levels=c(0:(num.classes - 1))))))
  conf.matrix <- rbind(conf.matrix[label,], colSums(conf.matrix[-(label),]))
  conf.matrix <- cbind(conf.matrix[,label], rowSums(conf.matrix[,-(label)]))
  return(conf.matrix)
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
  # shuffle data
  test <- test[sample(1:nrow(test)),]
  # compute iterator
  test.label <- test[, 2]
  test.iterator <- GetIterator(test, "test")
  
  # balance set
  eval <- BalanceSet(eval)
  # shuffle data
  eval <- eval[sample(1:nrow(eval)),]
  # compute iterator
  eval.iterator <- GetIterator(eval, "eval")
  
  # balance set
  train <- BalanceSet(train)
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

GetMetrics <- function(predicted, test.label, label){
  # computes all relevant evaluation metrics
  # get confusion matrix
  conf.matrix <- GetConfMatrix(predicted, test.label, label)
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

    # get ACC, SEV, SP
    label.metrics <- t(sapply(0:(num.classes - 1),
                              function(label)
                                GetMetrics(predicted, test.label, label)))

    # return classification metrics for each label (after one iteration)
    return(list("metrics" = label.metrics,
           "train.error" = training.result$train.error,
           "eval.error" = training.result$eval.error))
}

ReorderMatrixColumns <- function(m1, m2){
  # alternate columns from m1 and m2
  m3 <- NULL
  for (i in 1:ncol(m1)) m3 <- cbind(m3, m1[,i], m2[,i])
  return(m3)
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
  prefix <- sprintf("%s_%s_final_%s", dataset.name, method.name, dataset.type)
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
  plot(c(0, classifier.iterations), c(0, 100), type = "n",
       xlab = "Epochs", ylab = "Accuracy (%)",
       main = sprintf("(%s) - Validation results", dataset.name))
  x.axis <- seq(0, classifier.iterations,
                by = classifier.iterations / length(train.error))[-1]
  lines(x.axis, train.error * 100, col = "red", lwd = 2.5)
  eval.error <- c(rep(NA, each = (length(train.error) - length(eval.error))), eval.error)
  lines(x.axis, eval.error * 100, col = "blue", lwd = 2.5)
  legend(classifier.iterations / 2, 10,
         c("Training Accuracy (%)", "Evaluation Accuracy (%)"),
         lty = c(1, 1), lwd = c(2.5, 2.5), col = c("red", "blue"))
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
  
  # get method metrics
  method.metrics <- sapply(1:method.iterations,
                           function(x) matrix(unlist(method.result[x, 1]), ncol = 8),
                           simplify = FALSE)
  
  # return mean and variance of all iterations
  # METRIC1, VARIANCE2, METRIC2, VARIANCE2, ...
  metrics.means <- apply(simplify2array(method.metrics), 1:2, mean)
  metrics.vars <- apply(simplify2array(method.metrics), 1:2, var)
  return(ReorderMatrixColumns(metrics.means, metrics.vars))
}

GetMethodMetrics <- function(data.original){
  # get metrics for each label
  return(GetLabelMetrics(data.original))
}

MetricsStorage <- function(metrics){
  # create metrics dataframe
  metrics.data.frame <- data.frame(metrics)
  # assign column labels to data.frame
  names(metrics.data.frame) <- c("ACC", "Var(ACC)",
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

GetAllClassificationMetrics <- function(data.original){
  # get metrics for each image pre-processing method combination
  # all method metrics
  all.method.metrics <- GetMethodMetrics(data.original)

  # partial store of metrics
  MetricsStorage(all.method.metrics)
}