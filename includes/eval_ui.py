import sys
import os.path
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import posture
import cv2
import gui
import hand_detection
import fingertip
import time
def cutframe(frame):
    return frame[:380]

modelfilename = '../models/model1.mdl'
pos_recognizer = posture.PostureRecognizer.load(modelfilename)
ui = gui.GUI()
#Get the image and do the classification here
cap = cv2.VideoCapture(0)
def evaluation(filename = 'test.png', image =None):
    start = 0
    end = 0
    ui.canvas = np.ones((480,640,3L),)*255
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cutframe(frame)

    
        label, hand_mask = pos_recognizer.classify(frame)
        location, wrist_end = fingertip.find_fingertip(label, hand_mask)
        touching = posture.isTouching(frame, label, location, wrist_end)

        #=======The grammar goes here=============
        ui.handle_input(label, location, touching)

        cv2.imshow('Canvas', ui.get_screen())
        pressedKey = cv2.waitKey(60)
        if pressedKey == ord('q'):
            end = time.time()
            ui.save_canvas(filename)
            cv2.destroyAllWindows()
            return ui.canvas, end - start
        if pressedKey == ord('r'):
            # add something to add image 
            ui.draw_sample(image)
            start = time.time()
        if pressedKey & 0xFF == 27:
            break
            cv2.destroyAllWindows()
