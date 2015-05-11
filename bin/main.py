import sys
import os.path
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import posture
import cv2
import gui
import hand_detection
import fingertip
import hand_detection as hd
import numpy as np


def main():
    modelfilename = sys.argv[1]
    pos_recognizer = posture.PostureRecognizer.load(modelfilename)
    ui = gui.GUI()
    #Get the image and do the classification here
    cap = cv2.VideoCapture(1)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Display the input stream only for debug purposes
        cv2.imshow('input',frame)
        
        label, hand_mask, theta, skin_mask = pos_recognizer.classify(frame)
        print "label = ", label

        
        cv2.imshow('debug', frame[hand_mask.reshape(hand_mask.shape+(1,))] )
        # cv2.imshow('debug', np.rollaxis(np.array([frame[:,:,0]*hand_mask, frame[:,:,1]*hand_mask,frame[:,:,2]*hand_mask]),0,2) )

        if(label == posture.poses["UNKNOWN"]):
            continue
        location, wrist_end = fingertip.find_fingertip(label, skin_mask)
        if(not location):
            continue
        print "location= ", location
        print "wrist_end = ", wrist_end
        touching = posture.isTouching(frame, label, location, wrist_end)
        print "touching=" ,touching

        #=======The grammar goes here=============
        ui.handle_input(label, location, touching)

        cv2.imshow('Canvas', ui.get_screen())
        pressedKey = cv2.waitKey(60)


if __name__ == '__main__':
    main()