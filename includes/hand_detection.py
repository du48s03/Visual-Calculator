"""Package for hand detection"""
import cv2
import numpy as np
import pdb
import time
import math

def hand_detection(image):
    hsv_im = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    HSV = cv2.split(hsv_im)
    S = HSV[1]
    V = HSV[2]
    RGB = cv2.split(image)
    Blue = RGB[0]
    Green = RGB[1]
    Red = RGB[2]
    mask = Red - Blue
    rg = Red - Green
    mask[mask < 20] = 0
    mask[rg < 20] = 0
    mask[mask > 0] = 255
    mask[Blue > Red] = 0
    mask[Green > Red] = 0
    mask = cv2.medianBlur(mask,5)
    mask = np.bool_(mask)
    return mask
