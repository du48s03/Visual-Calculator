import numpy as np
import cPickle
import cv2
import os
import sys
import re

# Load the data set file.
#data and labels are lists of numpy.ndarray
filename = sys.argv[2]
print filename
if os.path.isfile(filename):
    inputfile = open(filename, 'r')
    old_data, old_labels, old_touchs = cPickle.load(inputfile)
    inputfile.close()
    # print "old_data = ", old_data.shape
    # print "old_label = ", old_labels.shape
else:
    old_data, old_labels, old_touchs = [],[],[]

inputfolder = sys.argv[1]
datalist, labellist, touchlist = [],[],[]

for f in os.listdir(inputfolder):
    img = cv2.imread(inputfolder+'/'+f)
    s = re.search(r'train_data_([0-9])_([0-9]+)\.png', f)
    label = s.group(1)
    n = s.group(2)
    print label, n
    # cv2.imshow(f, img)
    # cv2.waitKey(0)
    # label = raw_input("Enter the gesture:")
    # touch = raw_input("Does the finger touch the paper?")
    datalist.append(img)
    labellist.append(label)
    # touchlist.append(touch)

if len(datalist) != 0:
    data.extend(old_data)
    labels.extend(old_labels)
else:
    print "No data loaded!"
    exit()

if os.path.isfile(filename):
    os.remove(filename)
output = open(filename,'w')
cPickle.dump((data, labels), output)
output.close()
# When everything done, release the capture
