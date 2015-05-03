import sys
import os.path
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import posture
import cv2
import gui
import hand_detection
import fingertip

modelfilename = sys.argv[1]
pos_recognizer = posture.PostureRecognizer.load(modelfilename)
ui = gui.GUI()
#Get the image and do the classification here
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the input stream only for debug purposes
    cv2.imshow('input',frame)
    #pressedkey = cv2.waitKey(20) & 0xFF
    #if pressedkey == ord('q'):
    #    break

    
    label = pos_recognizer.classify(frame)
    location = fingertip.find_fingertip(frame, label)
    touching = posture.isTouching(frame, label, location)


    #=======The grammar goes here=============
    ui.setcursor(location)
    if posture == 1:
        ui.settool(gui.toolmap[1])
        if touching:
            #check the fingertip location to determine whether to draw or to change color/tool
            ui.drawpoint(location)
    elif posture == 2:
        #Don't have "selection" now, so implement this later. 
        # ui.settool(gui.toolmap[2])
        # ui.setcursor(location)
    elif posture == 3:
        ui.settool(gui.toolmap[3])
        if touching:
            ui.erase(location)
    else:#unknwon
        pass

    cv2.imshow('Canvas', ui.get_screen())
    pressedKey = cv2.waitKey(20)

