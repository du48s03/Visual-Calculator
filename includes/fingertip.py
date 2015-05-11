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
    # find the minimum / maximum x / y
    x_min = min(x)
    y_min = min(y)
    x_max = max(x)
    y_max = max(y)
    # find the center
    c_x = x.mean(axis=0)
    c_y = y.mean(axis=0)
    mask[:c_x,:] = 0
    # find the most largest edge (= wrist)
    up = (sum(mask[x_min,:])/255)
    bottom = (sum(mask[x_max,:])/255)
    left = (sum(mask[:,y_min])/255)
    right = (sum(mask[:,y_max])/255)
    print up,bottom,left,right
    wrist = max(up,bottom,left,right)
    wrist_ch = ''
    # mask wrist side
    if wrist == up:
        mask[:c_x,:] = 0
        wrist_ch = 'up'
    elif wrist == bottom:
        mask[c_x:,:] = 0
        wrist_ch = 'down'
    elif wrist == left:
        mask[:,:c_y] = 0
        wrist_ch = 'left'
    elif wrist == right:
        mask[:,c_y:] = 0
        wrist_ch = 'right'
    # find the most furthest point (= finger)
    (x,y) = mask.nonzero()
    z = (x - c_x)**2 + (y - c_y)**2
    if len(z) > 0:
        return x[z.argmax(0)],y[z.argmax(0)],wrist_ch
    else:
        return None, None, None

