import util
import sys
import cv2

img  = cv2.imread(sys.argv[1])
for g in range(100):
    print g/100.0
    binimg = util.getgreyat(img, g/100.0, r = 5)
    cv2.imshow('input', binimg)
    key = cv2.waitKey(0)