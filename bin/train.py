import sys
import os.path
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import posture
import cv2
import cPickle
import numpy as np

datasetname = sys.argv[1]
modelfilename = sys.argv[2]

print "Loading dataset..."
datafile = open(datasetname, 'r')
lists = cPickle.load(datafile)
data, labels = (lists[0], lists[1])
datafile.close()
print "Done\n"


# data_good = np.ones(len(labels), dtype=bool)
# for i in xrange(len(labels)):
#     if labels[i] not in posture.poses.values():
#         data_good[i] = False

# data = data[data_good]
# labels = labels[data_good]
# print "Good data = ", len(data)


if os.path.isfile(modelfilename):
    print "Loading: ", modelfilename
    pos_recognizer = posture.PostureRecognizer.load(modelfilename)
else:
    print "Creating: ", modelfilename
    pos_recognizer = posture.PostureRecognizer()
#Do the training here
print "Start training" 
pos_recognizer.train(data, labels)
print "saving data"
pos_recognizer.save(modelfilename)

