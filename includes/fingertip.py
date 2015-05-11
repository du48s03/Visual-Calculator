import cv2
import numpy as np
import pdb
import time
import math
def find_fingertip(label, mask):
    """
    Find the location of the fingertips. Only have to return one pixel. 
    @param np.array         hand_mask               a numpy array of size(480,640,1)
    @label string           label               A string represting the label of the frame. 
                                                Look at posture.py for more informations. 
    return (int, int)       loc                 The location of the fingertip of the longest finger.
    return string           wrist_end    "up/down/left/right" representing the side of the wrist. 
    """ 
    (x,y) = mask.nonzero()
    # find the center
    c_x = x.mean(axis=0)
    mask[:c_x,:] = 0
    # find the most furthest point (= finger)
    (x,y) = mask.nonzero()
    z = (x - c_x)**2 + (y - c_y)**2
    return x[z.argmax(0)],y[z.argmax(0)],wrist_ch

