import numpy as np
import hand_detection as hd
import majoraxis
import cv2
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib

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
        hand_mask, ang, skin_mask= hd.hand_detection(image)
        # while cv2.waitKey(20) != ord('a'):
        #     pass 
        # hand_mask_tmp = np.reshape(hand_mask, hand_mask.shape+(1,) )
        # hand_mask_tmp = np.concatenate((hand_mask_tmp,hand_mask_tmp,hand_mask_tmp),axis=2)

        image_gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #masked = np.multiply(image, hand_mask_tmp)
        masked = np.multiply(image_gray, hand_mask)
        plt.figure()
        plt.imshow(masked, cmap = matplotlib.cm.Greys_r)
        # plt.figure()
        # plt.plot(masked)
        # cv2.imshow('masked', masked)
        # while cv2.waitKey(20) != ord('a'):
        #     pass 


        #masked = image[hand_mask]
        sobelx = cv2.Sobel(masked,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(masked,cv2.CV_64F,0,1,ksize=5)

        angles = np.angle(sobelx + 1j*sobely,deg=True)
        # amplitudes = np.absolute(sobelx + 1j*sobely)
        # angles = angles.flatten()

        angles = angles.flatten()
        angles = angles[np.nonzero(angles)]

        # hist = cv2.calcHist(np.array([angles,amplitudes]).astype('float32'),\
        #     channels=[0,1],\
        #     mask=None,\
        #     histSize=[36,5],\
        #     # ranges=[[-180.0,180.0],[int(np.min(amplitudes)), int(np.max(amplitudes))]])
        #     ranges=[[-180.0,180.0],[0,256]])
        n_bins = 36
        # ang_hist, bins = np.histogram(angles, bins=n_bins,range=(-180.0,180.0))
        # ang_hist[18]=0
        # #print type(ang_hist)
        # amp_hist = np.histogram(amplitudes, bins=5)[0]
        # plt.figure()
        ang_hist,bins,patches = plt.hist(angles.flatten(), bins=n_bins, range=(-180, 180), normed=1)
        # print sum(ang_hist)
        


        #====Angular disposition=====
        # ang, cx, cy = majoraxis.majoraxis(hand_mask)
        ang = ang/pi*180
        try:
            majorangle = np.where(np.histogram([ang], bins=n_bins,range=(-180,180))[0] != 0)[0][0]
            ang_hist = np.roll(ang_hist, (n_bins - majorangle)%n_bins )
        except(IndexError):
            pass
        # plt.figure()
        # plt.hist(range(-180,180, 10), bins=n_bins, range=(-180,180), weights=ang_hist)
        # plt.show()
        return ang_hist, hand_mask, ang, skin_mask



        