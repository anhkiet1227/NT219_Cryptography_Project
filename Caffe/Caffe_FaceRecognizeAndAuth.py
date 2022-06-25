import numpy as np
import pickle
import os
import cv2
import time
import datetime
import imutils

#get the current directory of the file
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

#load the pickle file
print("Loading the pickle file")
recognizer = pickle.loads(open('pickleFile/recognizer.pickle', "rb").read())
le = pickle.loads(open('pickleFile/le.pickle', "rb").read())

#load the video stream
print("Starting recognition")
vs = cv2.VideoCapture(0)
time.sleep(1)

while True:
    #read the frame from the video stream
    ret, frame = vs.read()
    
    #resize the frame
    frame = imutils.resize(frame, width=600)

    #get the image dimensions
    (h, w) = frame.shape[:2]

    #detect the face with the face detector model - using ResNet Neural Network - Caffe Model
    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    #set the input to the face detector model
    face_detector.setInput(image_blob)
    
    #detect the face
    face_detections = face_detector.forward()

    for i in range(0, face_detections.shape[2]):
        
        #check if the face was detected
        confidence = face_detections[0, 0, i, 2]
        
        #check if the confidence is greater than the threshold
        if confidence >= 0.5:
            
            #get the coordinates of the face
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            
            #get the face ROI
            (startX, startY, endX, endY) = box.astype("int")
            
            #extract the face ROI
            face = frame[startY:endY, startX:endX]
            
            #get the face shape
            (fH, fW) = face.shape[:2]
            
            #resize the face ROI
            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)
        
            #set the input to the face recognition model
            face_recognizer.setInput(face_blob)
            
            #recognize the face
            vec = face_recognizer.forward()
            
            #get the label for the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            if(name == 'Unknown'):
                name = 'Unknown_Rejected'
            else:
                name = name + '_Accepted'

            #set the value of the text
            text = "{}: {:.2f}".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            #draw the rectangle around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            
            #draw the name of the person
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            
    #display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #if q is pressed, break the loop
    if key == ord('q'):
        break

#clean up the camera and close any open windows
cv2.destroyAllWindows()

#end of program
print("Finished")