import cPickle
import sys
import os
import cv2

inputfile  = open(sys.argv[1])
data, labels = cPickle.load(inputfile)
inputfile.close()

for i in xrange(len(data)):
    print labels[i]
    cv2.imshow('label = '+labels[i], data[i])
    while(cv2.waitKey(0) &0xFF != 27):
        pass
    cv2.destroyWindow('label = '+labels[i])
