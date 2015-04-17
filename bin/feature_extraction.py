import numpy as np
import hand_detection as hd
import cv2

class FeatureExtractor(object):
    """docstring for FeatureExtractor"""
    def __init__(self, arg):
        super(FeatureExtractor, self).__init__()
        self.arg = arg
        
    def extract_feature(self, image):
        """returns the feature vector of the image as a 1-D numpy array"""

class OrientationHistogramFeature(FeatureExtractor):
    """docstring for OrientationHistogramFeature"""
    def __init__(self, arg):
        super(OrientationHistogramFeature, self).__init__()
        self.arg = arg

    def extract_feature(self, image):
        hand_mask = hd.hand_detection(image)
        masked = np.multiply(image, hand_mask)
        sobelx = cv2.Sobel(masked,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(masked,cv2.CV_64F,0,1,ksize=5)

        angles = np.angle(sobelx + 1j*sobely,deg=True)
        amplitudes = np.absolute(sobelx + 1j*sobely)

        hist = nv2.calcHist([np.array([angles,amplitudes])],\
            channels=[0,1],\
            mask=None,\
            histSize=[36,5],\
            ranges=[[-180.0,180.0],[np.min(amplitudes), np.max(amplitudes)]])

        #ang_hist = np.histogram(angles, bins=36,range=(-180.0,180.0))
        #amp_hist = np.histogram(angles, bins=5)
        return hist[0].reshape(-1)



        