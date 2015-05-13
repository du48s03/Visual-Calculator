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

datafile = open(datasetname, 'r')
lists = cPickle.load(datafile)
data, labels = (lists[0], lists[1])
datafile.close()

# data_good = np.ones(len(labels), dtype=bool)
# for i in xrange(len(labels)):
#     if labels[i] not in posture.poses.values():
#         data_good[i] = False

# data = data[data_good]
# labels = labels[data_good]
# print "Good data = ", len(data)


if os.path.isfile(modelfilename):
    pos_recognizer = posture.PostureRecognizer.load(modelfilename)
else:
    pos_recognizer = posture.PostureRecognizer()
#Do the training here
print labels.size
pos_recognizer.train(data, labels)
pos_recognizer.save(modelfilename)

del data, labels

if(len(sys.argv) == 4):
    test_datasetname = sys.argv[3]
    testfile = open(test_datasetname, 'r')
    test_lists = cPickle.load(testfile)
    test_data, test_labels = (test_lists[0], test_lists[1])
    testfile.close()

    data_good = np.ones(len(test_labels), dtype=bool)

    score = 0.0
    total = 0.0
    for i in xrange(len(test_labels)):
        pred, hand_mask = pos_recognizer.classify(test_data[i])
        total+=1
        print i, 'pred = ', pred, 'label=' ,test_labels[i]
        if str(pred) == test_labels[i]:
            score +=1

    print "================\n"
    print "Score = ", score, "/", total, "=", score/total, "\n"

