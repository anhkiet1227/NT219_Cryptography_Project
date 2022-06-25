import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os
import imutils

#get the current working directory
curr_path = os.getcwd()

#load the face detection model
print("Loading face detection model")
proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

#load the face recognition model
print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

#load the database
print("Loading the database")
data_base_path = os.path.join(curr_path, 'database')

#load the labels
filenames = []
for path, subdirs, files in os.walk(data_base_path):
    for name in files:
        filenames.append(os.path.join(path, name))

#declare the list of labels
face_embeddings = []
face_names = []

for (i, filename) in enumerate(filenames):
    #load the image and process it
    print("Processing image {}".format(filename))

    #load the image
    image = cv2.imread(filename)
    
    #resize the image
    image = imutils.resize(image, width=600)

    #get the image dimensions
    (h, w) = image.shape[:2]

    #detect the face with the face detector model - using ResNet Neural Network - Caffe Model
    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    #set the input to the face detector model
    face_detector.setInput(image_blob)
    
    #detect the face
    face_detections = face_detector.forward()

    #check if at least one face was detected
    i = np.argmax(face_detections[0, 0, :, 2])
    
    #check if the face was detected
    confidence = face_detections[0, 0, i, 2]

    #check if the confidence is greater than the threshold
    if confidence >= 0.5:
        #get the coordinates of the face
        box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        
        #get the face ROI
        (startX, startY, endX, endY) = box.astype("int")

        #extract the face ROI
        face = image[startY:endY, startX:endX]

        #resize the face ROI
        face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0), True, False)

        #set the input to the face recognition model
        face_recognizer.setInput(face_blob)
        
        #recognize the face
        face_recognitions = face_recognizer.forward()

        #get the name of the label
        name = filename.split(os.path.sep)[-2]
        
        #add the embedding vector to the list
        face_embeddings.append(face_recognitions.flatten())
        
        #add the name to the list
        face_names.append(name)

#get the data with face embeddings and names
data = {"embeddings": face_embeddings, "names": face_names}

#get the encoded labels
le = LabelEncoder()

#transfer the names to the encoded labels
labels = le.fit_transform((data["names"]))

#get the recognizer
recognizer = SVC(C=1.0, kernel="linear", probability=True)

#get the recognizer with embeddings and labels
recognizer.fit(data["embeddings"], labels)

#save the recognizer to pickle file
print("Saving the pickle file")
f = open('pickleFile/recognizer.pickle', "wb")
f.write(pickle.dumps(recognizer))
f.close()

#save the label encoder to pickle file
f = open('pickleFile/le.pickle', "wb")
f.write(pickle.dumps(le))
f.close()

#end of the program
print("Finished")