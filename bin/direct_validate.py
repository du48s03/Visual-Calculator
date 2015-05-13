import numpy as np
import cPickle
import cv2
import os
import sys
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import re
import posture

# Load the data set file.
#data and labels are lists of numpy.ndarray



print "Loading Data"
inputfolder = sys.argv[1]
modelfilename = sys.argv[2]
datalist, labellist = [],[]


for f in os.listdir(inputfolder):
    s = re.search(r'train_data_([0-9])_([0-9]+)\.png', f)
    label = s.group(1)
    n = s.group(2)
    # print label, n
    if label == posture.poses["DBFINGER"]:
        continue
    img = cv2.imread(inputfolder+'/'+f)
    # cv2.imshow(f, img)
    # cv2.waitKey(0)
    # label = raw_input("Enter the gesture:")
    # touch = raw_input("Does the finger touch the paper?")
    datalist.append(img)
    labellist.append(label)
    # touchlist.append(touch)
print "Done\n"


#====Load the model==============
if os.path.isfile(modelfilename):
    print "Loading: ", modelfilename
    pos_recognizer = posture.PostureRecognizer.load(modelfilename)
else:
    raise ValueError("No model file "+modelfilename+" doesn't exists")
#Do the training here
# print "Start training" 
# pos_recognizer.train(traindata, trainlabels)
# print "saving data"
# pos_recognizer.save(modelfilename)


#==========validation===============
score = 0.0
total = 0.0
for i in xrange(len(datalist)):
    pred = pos_recognizer.classify(datalist[i])[0]
    total+=1
    print i, 'pred = ', pred, 'label=' ,labellist[i]
    if str(pred) == labellist[i]:
        score +=1

print "================\n"
print "Score = ", score, "/", total, "=", score/total, "\n"
