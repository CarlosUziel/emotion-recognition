require(gtools)

# Clean workspace
rm(list=ls())

# obtain the last image.needed images per subject capture
images.needed = 3

# set root dir
root.dir <- "/media/uziel/DATA/TFG/Data/CK+/"

###############################################################################
# GET SUBJECT-CAPTURE PAIRS WITH ONE IMAGE LABELED
###############################################################################

# look for all images with available labels
txt.files <- mixedsort(list.files(paste(root.dir, "Emotion/", sep = ""),
                                  pattern = "*.txt",
                                  recursive = TRUE))

# set images directory
images.dir <- paste(root.dir, "cohn-kanade-images/", sep = "")

# create directory of selected images
system(sprintf("rm -rf %s/SelectedImages", root.dir))
system(sprintf("mkdir %s/SelectedImages", root.dir))

# obtain labeled images
for (i in 1:length(txt.files)){
  # set subject and capture path
  capture.dir <- paste(images.dir, dirname(txt.files[i]), "/", sep = "")
  
  # look for all images of given subject and capture codes
  capture.images <- mixedsort(list.files(capture.dir,
                                         pattern = "*.png",
                                         recursive = TRUE))
  
  # get the images that must be selected
  if (images.needed == 0 || images.needed > length(capture.images)){
    selected <- round(length(capture.images) / 3)
    capture.images <- capture.images[-(1:(length(capture.images) - selected))]
  }else{
    capture.images <- capture.images[-(1:(length(capture.images) - images.needed))]
  }
  
  # store images
  store.dir <- paste(root.dir, "SelectedImages/", dirname(txt.files[i]), "/", sep = "")
  # create directories
  system(sprintf("mkdir %s -p", store.dir))
  
  # copy each image to new directory
  for (j in 1:length(capture.images)){
    image <- capture.images[j]
    system(sprintf("cp %s %s",
           paste(capture.dir, image, sep = ""),
           paste(store.dir, image, sep = "")))
  }
  print(sprintf("Done selecting subject: %d", i))
}

# obtain neutral face images
neutral.images <- mixedsort(list.files(images.dir,
                                  pattern = "*001.png",
                                  recursive = TRUE))

labels.dir <- paste(root.dir, "Emotion/", sep = "")

# copy each image to new directory
for (j in 1:length(neutral.images)){
  image <- neutral.images[j]
  store.dir <- paste(root.dir, "SelectedImages/", sep = "")
  system(sprintf("mkdir %s -p", paste(store.dir, dirname(image), sep = "")))
  system(sprintf("cp %s %s",
                 paste(images.dir, image, sep = ""),
                 paste(store.dir, image, sep = "")))
  print(sprintf("Done copying neutral face: %d", j))
}


###############################################################################
# CROP LABELED IMAGES
###############################################################################

# crop the faces from all the selected images

# set images directory
source.dir <- paste(root.dir, "SelectedImages/", sep = "")
destination.dir <- paste(root.dir, "CroppedImages/", sep = "")

# set crop script dir
crop.script.path = "/media/uziel/DATA/TFG/FaceRecognition/crop_face.py"

# get all selected images
images.files <- mixedsort(list.files(source.dir,
                                     pattern = "*.png",
                                     recursive = TRUE))

# create directory of cropped images
system(sprintf("rm -rf %sCroppedImages", root.dir))
system(sprintf("mkdir %sCroppedImages", root.dir))

# obtain cropped images
for (i in 1:length(images.files)){
  # set origin path
  origin.path <- paste(source.dir, images.files[i], sep = "")
  # set store path
  store.path <- paste(destination.dir, images.files[i], sep = "")
  
  # create store directories
  # get subject and capture codes
  store.dir <- paste(destination.dir, dirname(images.files[i]), sep = "")
  system(sprintf("mkdir %s -p", store.dir))
  
  # crop image
  system(sprintf("python %s %s %s", crop.script.path, origin.path, store.path))
  
  print(sprintf("Done cropping image: %d", i))
}

###############################################################################
# AUGMENT ORIGINAL DATA FROM CROPPED IMAGES
###############################################################################
# https://www.bioconductor.org/packages/devel/bioc/vignettes/EBImage/inst/doc/EBImage-introduction.html
require(EBImage)
require(tools)

#TODO
# read each image, write augmented copies
images.dir <- paste(root.dir, "CroppedImages/", sep = "")
images.files <- mixedsort(list.files(images.dir,
                                     pattern = "*.png",
                                     recursive = TRUE))

for(i in 1:length(images.files)){
  # load image
  image <- readImage(paste(images.dir, images.files[i], sep = ""))
  # apply transformations to image
  
  # flip image horizontally (vertical axis)
  writeImage(flop(image), paste(images.dir, file_path_sans_ext(images.files[i]), "-flop", ".png", sep = ""))
  
  #adjust brightness
  
  writeImage(image + 0.2, paste(images.dir, file_path_sans_ext(images.files[i]), "+b", ".png", sep = ""))
  writeImage(image - 0.2, paste(images.dir, file_path_sans_ext(images.files[i]), "-b", ".png", sep = ""))
  
  #adjust contrast
  writeImage(image * 1.5, paste(images.dir, file_path_sans_ext(images.files[i]), "+c", ".png", sep = ""))
  writeImage(image * 0.5, paste(images.dir, file_path_sans_ext(images.files[i]), "-c", ".png", sep = ""))
  
  #adjust gamma
  writeImage(image ^ 1.5, paste(images.dir, file_path_sans_ext(images.files[i]), "+g", ".png", sep = ""))
  writeImage(image ^ 0.5, paste(images.dir, file_path_sans_ext(images.files[i]), "-g", ".png", sep = ""))
  
  # translate
  image.translate = translate(image, c(50, 0))
  writeImage(image.translate, paste(images.dir, file_path_sans_ext(images.files[i]), "+ht", ".png", sep = ""))
  image.translate = translate(image, c(-50, 0))
  writeImage(image.translate, paste(images.dir, file_path_sans_ext(images.files[i]), "-ht", ".png", sep = ""))
  image.translate = translate(image, c(0, 50))
  writeImage(image.translate, paste(images.dir, file_path_sans_ext(images.files[i]), "+vt", ".png", sep = ""))
  image.translate = translate(image, c(0, -50))
  writeImage(image.translate, paste(images.dir, file_path_sans_ext(images.files[i]), "-vt", ".png", sep = ""))
  
  # rotate and crop
  image.rotate = rotate(image, 20, bg.col = "black")
  image.resize = resize(image.rotate, w = nrow(image))
  writeImage(image.resize, paste(images.dir, file_path_sans_ext(images.files[i]), "+20r", ".png", sep = ""))
  image.rotate = rotate(image, 10, bg.col = "black")
  image.resize = resize(image.rotate, w = nrow(image))
  writeImage(image.resize, paste(images.dir, file_path_sans_ext(images.files[i]), "+10r", ".png", sep = ""))
  image.rotate = rotate(image, -10, bg.col = "black")
  image.resize = resize(image.rotate, w = nrow(image))
  writeImage(image.resize, paste(images.dir, file_path_sans_ext(images.files[i]), "-10r", ".png", sep = ""))
  image.rotate = rotate(image, -20, bg.col = "black")
  image.resize = resize(image.rotate, w = nrow(image))
  writeImage(image.resize, paste(images.dir, file_path_sans_ext(images.files[i]), "-20r", ".png", sep = ""))
  
  print(sprintf("Done augmenting image: %d", i))
  
}

###############################################################################
# GET IMAGES LABELS
###############################################################################
# make labels file
# set labels dir
labels.dir <- paste(root.dir, "Emotion/", sep = "")
# set images dir
images.dir <- paste(root.dir, "CroppedImages/", sep = "")
images.files <- mixedsort(list.files(images.dir,
                                     pattern = "*.png",
                                     recursive = TRUE))

# get neutral face labels
neutral.images <- mixedsort(list.files(images.dir,
                                       pattern = "*00000001*",
                                       recursive = TRUE))

# substract neutral from the rest
images.files <- setdiff(images.files, neutral.images)

GetLabel <- function(x, labels.dir, file){
  image.label.dir <- paste(labels.dir, dirname(file), "/", sep = "")
  label.file <- list.files(image.label.dir,
                           pattern = ".txt")
  label <- scan(paste(image.label.dir, label.file, sep = ""))
  return(c(x, as.integer(label), file))
}

# get label for each file
labeled.records <- t(sapply(1:length(images.files), function(x) GetLabel(x, labels.dir, images.files[x])))

# set neutral label to neutral face images
neutral.index <- c((nrow(labeled.records) + 1):(nrow(labeled.records) + length(neutral.images)))
neutral.labels <- rep(0, length(neutral.images))
neutral.records <- cbind(neutral.index, neutral.labels, neutral.images)

records <- rbind(labeled.records, neutral.records)

# write record
write.table(  records,
              file = paste(root.dir, "ck_extended_record.csv", sep = ""), 
              sep = ",",
              row.names = FALSE, 
              col.names = FALSE,
              quote = FALSE)