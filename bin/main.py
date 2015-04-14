import sys
import os.path
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import posture
import cv2


modelfilename = sys.argv[1]
pos_recognizer = posture.PostureRecognizer()

#Get the image and do the classification here