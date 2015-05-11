from sklearn import svm, neighbors
import numpy as np
import pickle
import os
import hand_detection
import feature_extraction as fe
import cv2
from collections import deque

poses = {
'POINTING':'1',
'DBFINGER':'2',
'PALM':'3',
'UNKNOWN':'-1',
}
## 1 = pointing finger
## 2 = piece sign
## 3 = palm
## -1 = unknown



class PostureRecognizer(object):
    """A class for hand posture recognition"""
    def __init__(self):
        super(PostureRecognizer, self).__init__()
        self.feature_extractor = fe.OrientationHistogramFeature()
        self.classifier = neighbors.KNeighborsClassifier()
        self.model = None


    def save(self, filename):
        """Save the current trained model to a pickle file.

        params filename: The output filename. 
        """
        if os.path.isfile(filename):
            os.remove(filename)
        outputfile = open(filename,'w')
        pickle.dump(self,outputfile)
        outputfile.close()

    @staticmethod
    def load(filename):
        """load a trained model from the filename.

        parmas  filename: The filename of the trained model
        return  model: A PostureRecognizer instance. """
        modelfile = open(filename, 'r')
        self = pickle.load(modelfile)
        modelfile.close()
        return self

    def extract_features(self, image):
        """Extract the features from the image"""
        return self.feature_extractor.extract_features(image)


    def train(self, train_data, train_label):
        """train on the training data and the labels. The train_data is a matrix of n x m x 3, 
        where each row is the features of one example. The train_label is a vector of n x 1. 

        params train_data: the training data, a numpy.ndarray. 
        params train_label: the training labels, a numpy.ndarray. """
        #Get the dimension of the feature first

        img = train_data[0,:,:,:]
        feature= self.extract_features(img)[0]
        features = np.zeros((train_data.shape[0], feature.size))
        for i in xrange(train_data.shape[0]):
            label = train_label[i]
            img = train_data[i,:,:,:]
            feature= self.extract_features(img)[0]
            features[i,:] = feature
        
        self.classifier.fit(features,train_label)

    def hand_detection(self, image):
        """transfer the input image into a binary matrix which marks where the hand is. Each pixel 
        of the output is Ture if and only if the corresponding pixel in the original image is in the 
        hand area. 

        params image: A numpy.ndarray representing the input image taken with cv2.imread() in BGR mode. 
        return mask: A numpy.ndarray with the same shape with the input image. 
        """
        return hand_detection.hand_detection(self, image)

    def classify(self, image):
        """Classify the image into one of the defined postures.
        
        params image: A numpy.ndarray representing the input image taken with cv2.imread() in BGR mode. 
        params hand_mask: A numpy.ndarray with the same shape with the input image which indicate where the hand is. 
        return posture :A string which corresponds to one of the defined postures, or -1 if unknown"""
        feature, hand_mask, theta, skin_mask = self.extract_features(image)
        pred = self.classifier.predict(feature)
        #TODO return -1 if dist is too large
        if pred not in poses.values():
            return poses["UNKNOWN"], hand_mask, theta, skin_mask
        return pred[0], hand_mask, theta, skin_mask

def find_shadow_of(img, location, hand_mask):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_B = (img[:,:,0] > 0.21 * 180) * (img[:,:,0] < 0.67*180)
    mask_G = (img[:,:,1] > 0.1 *255) * (img[:,:,1] < 0.34*255)
    mask_R = (img[:,:,2] > 0 *255) * (img[:,:,2] < 0.42*255)
    
    # mask_H = (img_hsv[:,:,0] > 0.21 * 180) * (img_hsv[:,:,0] < gr*180)
    # mask_S = (img_hsv[:,:,1] > gray_val * 180) * (img_hsv[:,:,1] < 1*180)
    mask_V = (img_hsv[:,:,2] > 0.0 * 180) * (img_hsv[:,:,2] < 0.56*180)
    mask_nothand = hand_mask == False

    mask = mask_R*mask_G*mask_B*mask_V*mask_nothand
    mask[location[0]:,:] = False
    mask[:,location[1]+40:] = False


    # for i in xrange(binimg.shape[0]):
    #     for j in xrange(binimg.shape[1]):
    #         if i>location[0] or j < location[1] or j > location[1] + 40:
    #             binimg[i,j]= False

    return mask

def shadow_fingertip(shadow_mask, wrist_end):
    shadow_finger = None
    shadow_indices_i, shadow_indices_j = np.where(shadow_mask)
    if len(shadow_indices_i) ==0:
        return shadow_finger

    if wrist_end == 'up':
        finger_ind = shadow_indices_i.argmax()
    elif wrist_end == 'down':
        finger_ind = shadow_indices_i.argmin()
    elif wrist_end == 'left':
        finger_ind = shadow_indices_j.argmax()
    elif wrist_end == 'right':
        finger_ind = shadow_indices_j.argmin()
    else:
        raise ValueError('Unknown wrist_end')
    shadow_finger = (shadow_indices_i[finger_ind], shadow_indices_j[finger_ind])
    return shadow_finger

def isTouching(frame, label, location, wrist_end, hand_mask):
    """
    Determin if the finger is touching the paper. 
    """
    shadow_mask = find_shadow_of(frame,location, hand_mask)
    shadow_ft = shadow_fingertip(shadow_mask, wrist_end)
    # frame_tmp = np.copy(frame)
    # frame_tmp[shadow_mask==False]=0
    # cv2.circle(frame_tmp,shadow_ft ,3,(0,0,255),-1)
    # cv2.imshow('shadow_fingertip', np.multiply(frame, shadow_mask))
    frame_tmp = np.copy(frame)
    frame_tmp[shadow_mask==False]=0
    cv2.circle(frame_tmp,shadow_ft ,3,(0,0,255),-1)
    cv2.imshow('shadow', frame_tmp)

    print shadow_ft
    if shadow_ft is None:
        return False

    return (location[0]-shadow_ft[0])**2 + (location[1]-shadow_ft[1])**2 < 100

        




        