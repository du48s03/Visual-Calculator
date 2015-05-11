import numpy as np
import cv2

def getgreyat(img, gray_val, r = 5):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_B = (img[:,:,0] > 0.21 * 180) * (img[:,:,0] < 0.67*180)
    mask_G = (img[:,:,1] > 0.1 *255) * (img[:,:,1] < 0.34*255)
    mask_R = (img[:,:,2] > 0 *255) * (img[:,:,2] < 0.42*255)
    
    # mask_H = (img_hsv[:,:,0] > 0.21 * 180) * (img_hsv[:,:,0] < gr*180)
    # mask_S = (img_hsv[:,:,1] > gray_val * 180) * (img_hsv[:,:,1] < 1*180)
    mask_V = (img_hsv[:,:,2] > 0.0 * 180) * (img_hsv[:,:,2] < 0.56*180)
    
    mask = mask_R*mask_G*mask_B*mask_V
    mask[300:,:] = False

    

    #binimg = np.rollaxis(np.array([img_hsv[:,:,0], img_hsv[:,:,0], img_hsv[:,:,0]]), 0,3).astype(np.uint8)*255
    return mask
    #return img_hsv


