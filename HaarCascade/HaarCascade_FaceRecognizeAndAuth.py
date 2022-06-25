import face_recognition
import imutils
import pickle
import time
import cv2
import os
from imutils.video import VideoStream

# load the detection model
print("Loading the detection model")
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt.xml"

# load the recognition model
print("Loading the recognition model")
faceCascade = cv2.CascadeClassifier(cascPathface)

# load the pickle file
print("Loading the pickle file")
data = pickle.loads(open('pickleFile/face.pickle', "rb").read())

# start the video stream
print("Starting recognition")
video_capture = cv2.VideoCapture(0)

while True:

    # get the frame from the threaded video stream
    ret, frame = video_capture.read()

    # convert the frame to grayscale and resize it to the width of the screen
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # transfer the input frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # initialize the list of recognized faces
    encodings = face_recognition.face_encodings(rgb)
    names = []

    for encoding in encodings:

        # match the encoding against the known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)

        # set name = unknown and reject if there are too many or too few matches or no encoding matches
        name = "Unknown_Rejected"

        # if the encoding matches, set the name to the name of the person
        if True in matches:

            # get the index of the first match
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]

            # count the number of matches for each person
            counts = {}

            for i in matchedIdxs:

                # get the name of the person
                name = data["names"][i]

                # increase count for the name we got
                counts[name] = counts.get(name, 0) + 1

            # set name which has highest count
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

        for ((x, y, w, h), name) in zip(faces, names):

            # draw the rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # draw the name of the person
            if (name != "Unknown_Rejected"):
                cv2.putText(frame, name + "_Accepted", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 225, 0), 2)
            else:
                cv2.putText(frame, name, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 225, 0), 2)

    # display the frame
    cv2.imshow("Frame", frame)

    # if q is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean up the camera and close any open windows
video_capture.release()
cv2.destroyAllWindows()

# end of program
print("Finished")
