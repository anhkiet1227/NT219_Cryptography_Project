from imutils import paths
import face_recognition
import pickle
import cv2
import os
 
#get the data from Images
imagePaths = list(paths.list_images('Images'))
knownEncodings = []
knownNames = []

#start processing the images
for (i, imagePath) in enumerate(imagePaths):
    print("Processing image {}".format(imagePaths)) 
    
    #get the name of the person from the image
    name = imagePath.split(os.path.sep)[-2]
    
    #get the picture and transfer to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #use face_recognition to get the face location
    boxes = face_recognition.face_locations(rgb,model='hog')
    
    #get the facial of the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    #save the data to the knownEncodings and knownNames
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        
#save the encodings and names to data
data = {"encodings": knownEncodings, "names": knownNames}

#use pickle to save data into face.pickle and use to recognize
f = open('pickleFile/face.pickle', "wb")
f.write(pickle.dumps(data))
f.close()

#end of the program
print("Finished")