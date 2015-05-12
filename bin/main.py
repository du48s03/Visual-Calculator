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

def mouse_callback(event,x,y,flags,param):
    i,j = (y-1,x-1)
    global counter
    if event == cv2.EVENT_LBUTTONUP:
        print i,j


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
        # print "label = ", label

        
        #cv2.imshow('debug', frame[)] )
        frame_tmp = np.copy(frame)
        frame_tmp[hand_mask==False] = 0
        cv2.namedWindow('debug')
        cv2.setMouseCallback('debug',mouse_callback)
        cv2.imshow('debug', frame_tmp)

        if(label == posture.poses["UNKNOWN"]):
            continue
        location, wrist_end = fingertip.find_fingertip(label, skin_mask)
        wrist_end = 'up'
        if(not location):
            continue
        # print "location= ", location
        # print "wrist_end = ", wrist_end
        touching = posture.isTouching(frame, label, location, wrist_end, hand_mask)
        # print "touching=" ,touching

        #=======The grammar goes here=============
        print "label = ", label, "location", location, "touching", touching
        ui.handle_input(label, location, touching)
        cv2.imshow('Canvas', ui.get_screen())
        

        pressedKey = cv2.waitKey(60)
        if pressedKey == 27:
            break

if __name__ == '__main__':
    main()