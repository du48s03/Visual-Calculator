from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import numpy as np
import pickle
import os
import hand_detection
import feature_extraction as fe

class PostureRecognizer(object):
    """A class for hand posture recognition"""
    def __init__(self):
        super(PostureRecognizer, self).__init__()
        self.feature_extractor = fe.OrientationHistogramFeature()
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
        return self.feature_extractor.extract_features(image)


    def train(self, train_data, train_label):
        """train on the training data and the labels. The train_data is a matrix of n x m x 3, 
        where each row is the features of one example. The train_label is a vector of n x 1. 

        params train_data: the training data, a numpy.ndarray. 
        params train_label: the training labels, a numpy.ndarray. """
        #Get the dimension of the feature first
        img = train_data[0,:,:,:]
        feature = self.extract_features(img)
        features = np.zeros(train_data.shape[0], feature.size)
        for i in xrange(train_data.shape[0]):
            img = train_data[i,:,:,:]
            feature = self.extract_features(image)
            label = train_label[i]
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
        return posture :An int which corresponds to one of the defined postures"""
        feature = self.extract_features(image)
        pred = self.classifier.predict(feature)
        return int(pred[0])

        




        