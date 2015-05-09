import numpy as np
import cv2

def getgreyat(img, gray_val, r = 5):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_H = (img[:,:,0] > 0.29 * 180) * (img[:,:,0] < 0.57*180)
    mask_S = (img[:,:,1] > 0.19 *255) * (img[:,:,1] < 0.30*255)
    mask_V = (img[:,:,2] > 0.14*255) * (img[:,:,2] < 0.23*255)
    mask = mask_H*mask_S*mask_V
    binimg = np.rollaxis(np.array([mask, mask, mask]), 0,3).astype(np.uint8)*255
    #binimg = np.rollaxis(np.array([img_hsv[:,:,0], img_hsv[:,:,0], img_hsv[:,:,0]]), 0,3).astype(np.uint8)*255
    return binimg*img
    #return img_hsv


