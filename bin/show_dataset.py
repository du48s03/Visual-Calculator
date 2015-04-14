import cPickle
import sys
import os
import cv2
import numpy as np

inputfile  = open(sys.argv[1])
data, labels = cPickle.load(inputfile)
inputfile.close()

for i in xrange(data.shape[0]):
    print "index = ", i, "label = ", int(labels[i])
    cv2.imshow('label = '+str(int(labels[i])), data[i,:,:,:])
    while(cv2.waitKey(0) &0xFF != 27):
        pass
    cv2.destroyWindow('label = '+str(int(labels[i])))
