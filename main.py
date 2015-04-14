import posture
import cv2
import sys

modelfilename = sys.argv[1]
pos_recognizer = posture.PostureRecognizer()
pos_recognizer.save(modelfilename)
pos_recognizer = posture.PostureRecognizer.load(modelfilename)


