import cv2
import sys

# set cascPath
cascPath = "/media/uziel/DATA/TFG/FaceRecognition/haarcascade_frontalface_default.xml"
# Get user supplied values
imagePath = sys.argv[1]
storePath = sys.argv[2]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=1,
    minSize=(30, 30)
)

# save cropped face
for (x, y, w, h) in faces:
    cv2.imwrite(storePath, image[y:(y+h), x:(x+w)])
