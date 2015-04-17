import sys
import os.path
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import posture
import cv2
import cPickle

datasetname = sys.argv[1]
modelfilename = sys.argv[2]

datafile = open(datasetname, 'r')
data, labels = cPickle.load(datafile)
datafile.close()


if os.path.isfile(modelfilename):
    pos_recognizer = posture.PostureRecognizer.load(modelfilename)
else:
    pos_recognizer = posture.PostureRecognizer()
#Do the training here
print labels.size
pos_recognizer.train(data, labels)
pos_recognizer.save(modelfilename)

