import util
import sys
import cv2
import numpy as np

img  = cv2.imread(sys.argv[1])
cap = cv2.VideoCapture(1)
while(True):
    ret, img = cap.read()
    for g in range(100,-1,-1):
        print g/100.0
        mask = util.getgreyat(img, g/100.0)
        frame_tmp = np.copy(img)
        frame_tmp[mask==False]=0
        # cv2.circle(frame_tmp,shadow_ft ,3,(0,0,255),-1)
        cv2.imshow('shadow', frame_tmp)
        key = cv2.waitKey(0)

