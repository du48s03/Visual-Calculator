"""Package for hand detection"""
import cv2
import numpy as np
import pdb
import time
import math
import majoraxis

def detectwrist(mask):
    (x,y) = mask.nonzero()
    # find the minimum / maximum x / y
    x_min = min(x)
    y_min = min(y)
    x_max = max(x)
    y_max = max(y)
    # find the center
    c_x = x.mean(axis=0)
    c_y = y.mean(axis=0)
    l = np.where(mask[:,y_min])[0][0]
    r = np.where(mask[:,y_max])[0][0]
    slopes = []
    if l > r:
        x_2 = r
        y_2 = sum(mask[r,:])
    else:
        x_2 = l
        y_2 = sum(mask[l,:])
    # if the condition is not good, simply half it
    if x_2 <= x_min + 30:
        mask[x_min:x_min+int(x_max-x_min)/2,:] = 0
        return mask
    # use the method from paper[5].
    else:
        for i in range(x_min,x_2-30):
            slope = (y_2 - (sum(mask[i,:])/255))*1.0/(x_2 - i)
            slopes.append(slope)
        wrist = slopes.index(max(slopes))
        mask[x_min:x_min+wrist,:] = 0
    return mask

# in order to detect wrist correctly, rotate image before detecting wrist
def rotateim(im,theta,c_x,c_y):
    c_x = int(c_x)
    c_y = int(c_y)
    rows,cols = im.shape[:2]
    dst = im
    M = np.float32([[1,0,cols/2-c_y],[0,1,rows/2 -c_x]])
    dst = cv2.warpAffine(im,M,(cols,rows))
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-(theta*180/math.pi),1)
    dst = cv2.warpAffine(dst,M,(cols,rows))
    return dst
# skin color detection using hsv,rgb
def skin_color(image):
    # detect hand region
    hsv_im = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    HSV = cv2.split(hsv_im)
    S = HSV[1]
    V = HSV[2]
    RGB = cv2.split(image)
    Blue = RGB[0]
    Green = RGB[1]
    Red = RGB[2]
    mask = Red - Blue
    mask[mask > 0] = 255
    mask[Blue > Red] = 0
    mask[Green > Red] = 0
    mask[V > 0.73*255] = 0
    mask[S < 0.3*255] = 0
    # desk zone
    #mask[:,:100] = 0
    #mask[376:,:]=0
    # erase spot noise
    mask = cv2.medianBlur(mask,15)
    return mask
# main function -- detecting hand region
def hand_detection(image):
    # skin color detection from the original image
    mask = skin_color(image)
    # detect the major axis angle
    [theta,c_x,c_y] = majoraxis.majoraxis(mask)
    image[mask==False] = 0
    cv2.imshow('test',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # rotate image and detect wrist
    mask = rotateim(mask,theta,c_x,c_y)
    mask = detectwrist(mask)
    mask = np.bool_(mask)
    return mask
