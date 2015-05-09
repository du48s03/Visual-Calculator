import numpy as np
import hand_detection as hd
import majoraxis
import cv2
from numpy import pi

class FeatureExtractor(object):
    """docstring for FeatureExtractor"""
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
    def extract_feature(self, image):
        """returns the feature vector of the image as a 1-D numpy array"""

class OrientationHistogramFeature(FeatureExtractor):
    """docstring for OrientationHistogramFeature"""
    def __init__(self):
        super(OrientationHistogramFeature, self).__init__()
 

    def extract_features(self, image):
        hand_mask = hd.hand_detection(image)
        #hand_mask = image[2] > 40.0
        # hand_mask_show = hand_mask.astype(np.int)*255*255
        # cv2.imshow('hand_mask', hand_mask.astype(np.int)*255*255)
        # while cv2.waitKey(20) != ord('a'):
        #     pass 
        hand_mask_tmp = np.reshape(hand_mask, hand_mask.shape+(1,) )
        hand_mask_tmp = np.concatenate((hand_mask_tmp,hand_mask_tmp,hand_mask_tmp),axis=2)

        masked = np.multiply(image, hand_mask_tmp)
        # cv2.imshow('masked', masked)
        # while cv2.waitKey(20) != ord('a'):
        #     pass 
        # exit()

        #masked = image[hand_mask]
        sobelx = cv2.Sobel(masked,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(masked,cv2.CV_64F,0,1,ksize=5)

        angles = np.angle(sobelx + 1j*sobely,deg=True)
        amplitudes = np.absolute(sobelx + 1j*sobely)

        # hist = cv2.calcHist(np.array([angles,amplitudes]).astype('float32'),\
        #     channels=[0,1],\
        #     mask=None,\
        #     histSize=[36,5],\
        #     # ranges=[[-180.0,180.0],[int(np.min(amplitudes)), int(np.max(amplitudes))]])
        #     ranges=[[-180.0,180.0],[0,256]])
        n_bins = 36
        ang_hist, bins = np.histogram(angles, bins=n_bins,range=(-180.0,180.0))
        #print type(ang_hist)
        amp_hist = np.histogram(amplitudes, bins=5)[0]

        #====Angular disposition=====
        ang, cx, cy = majoraxis.majoraxis(hand_mask)
        ang = ang/pi*180
        majorangle = np.where(np.histogram([ang], bins=n_bins,range=(-180,180))[0] != 0)[0][0]
        ang_hist = np.roll(ang_hist, (n_bins - majorangle)%n_bins )

        return ang_hist, hand_mask



        