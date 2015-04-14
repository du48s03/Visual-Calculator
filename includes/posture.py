from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import os


class PostureRecognizer(object):
    """A class for hand posture recognition"""
    def __init__(self):
        super(PostureRecognizer, self).__init__()
        #self.feature_extractor = FeatureExtractor()
        self.classifier = NearestNeighbors()
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


    def train(self, train_data, train_label):
        """train on the training data and the labels. The train_data is a matrix of n-by-d, 
        where each row is the features of one example. The train_label is a vector of n-by-1. 

        params train_data: the training data, a numpy.ndarray. 
        params train_label: the training labels, a numpy.ndarray. """
        pass

    def hand_detection(self, image):
        """transfer the input image into a binary matrix which marks where the hand is. Each pixel 
        of the output is Ture if and only if the corresponding pixel in the original image is in the 
        hand area. 

        params image: A numpy.ndarray representing the input image taken with cv2.imread() in BGR mode. 
        return mask: A numpy.ndarray with the same shape with the input image. 
        """
        pass

    def posture_recognition(self, image, hand_mask):
        """Classify the image into one of the defined postures.
        
        params image: A numpy.ndarray representing the input image taken with cv2.imread() in BGR mode. 
        params hand_mask: A numpy.ndarray with the same shape with the input image which indicate where the hand is. 
        return posture :An int which corresponds to one of the defined postures"""

        




        