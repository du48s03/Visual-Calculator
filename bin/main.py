import sys
import os.path
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import posture
import cv2


modelfilename = sys.argv[1]
pos_recognizer = posture.PostureRecognizer.load(modelfilename)

#Get the image and do the classification here
cap = cv2.VideoCapture(0)

expression = ''
s1 = 0
s2 = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('frame',frame)
    pressedkey = cv2.waitKey(20) & 0xFF
    if pressedkey == ord('q'):
        break
    elif pressedkey == ord('z'): 
        #take one shot
        posture = pos_recognizer.classify(frame)
        print posture
