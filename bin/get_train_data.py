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
    old_data, old_labels = cPickle.load(inputfile)
    inputfile.close()
    # print "old_data = ", old_data.shape
    # print "old_label = ", old_labels.shape
else:
    old_data, old_labels = None, None


cap = cv2.VideoCapture(0)

datalist, labellist = [],[]

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
        label = raw_input("What's the label of this picture?")
        if len(label) != 0:
            datalist.append(frame)
            labellist.append(label)

if len(datalist) != 0:
    newdata = np.array(datalist)
    newlabels = np.array(labellist)

    data = newdata if old_data is None else np.concatenate((old_data, newdata),axis=0)
    labels = newlabels if old_labels is None else np.concatenate((old_labels, newlabels),axis=0)

if os.path.isfile(filename):
    os.remove(filename)
output = open(filename,'w')
cPickle.dump((data, labels), output)
output.close()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
