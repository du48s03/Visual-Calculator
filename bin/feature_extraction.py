import numpy as np
import hand_detection as hd
import cv2

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
        # cv2.imshow('hand_mask', hand_mask.astype(int)*128)
        # cv2.waitKey(0)
        hand_mask = np.reshape(hand_mask, hand_mask.shape+(1,) )
        hand_mask = np.concatenate((hand_mask,hand_mask,hand_mask),axis=2)
        masked = np.multiply(image, hand_mask)
        # cv2.imshow('testwindow', masked)
        # cv2.waitKey( 0)
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
        ang_hist = np.histogram(angles, bins=36,range=(-180.0,180.0))[0]
        #print type(ang_hist)
        amp_hist = np.histogram(angles, bins=5)[0]
        return ang_hist



        