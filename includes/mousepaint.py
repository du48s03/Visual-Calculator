import os
import sys
import gui
import cv2
import numpy as np
import time

global ui 
ui = gui.GUI()

global mode 
global isTouching
isTouching = False
mode = '1'

def mouse_cb(event,x,y,flags,param):
    global ui
    global mode
    global isTouching
    if(event == cv2.EVENT_LBUTTONDOWN):
        isTouching = True
    if(event == cv2.EVENT_LBUTTONUP):
        isTouching = False

    ui.handle_input(mode, (x,y), isTouching)

def evaluation(filename = 'test.png',image = None):
    cv2.namedWindow('gui')
    cv2.setMouseCallback('gui', mouse_cb)
    start = 0
    end = 0
    while(True):
        cv2.imshow('gui',ui.get_screen())
        pressedKey = cv2.waitKey(60)

        if pressedKey == ord('r'):
            ui.draw_sample(image)
            start = time.time()
        if pressedKey == ord('q'):
            end = time.time()
            ui.save_canvas(filename)
            return ui.canvas, end - start
        if pressedKey & 0xFF == 27:
            break
            cv2.destroyAllWindows()

