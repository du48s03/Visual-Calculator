import numpy as np
import cPickle
import cv2
import os
import sys

# Load the data set file.
#data and labels are lists of numpy.ndarray
filename = sys.argv[1]
print filename
if os.path.isfile(filename):
    inputfile = open(filename, 'r')
    data, labels = cPickle.load(filename)
    inputfile = close()
else:
    data, labels = [], []

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    pressedkey = cv2.waitKey(1) & 0xFF
    if pressedkey == ord('q'):
        break
    elif pressedkey == ord('z'): 
        #take one shot
        label = raw_input("What's the label of this picture?")
        if len(label) != 0:
            data.append(frame)
            labels.append(label)

if os.path.isfile(filename):
    os.remove(filename)
output = open(filename,'w')
cPickle.dump((data, labels), output)
output.close()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
