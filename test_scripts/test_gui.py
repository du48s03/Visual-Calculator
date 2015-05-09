import os
import sys
includespath = os.path.abspath('../includes')
sys.path.insert(0, includespath)
import gui
import cv2
import numpy as np


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

cv2.namedWindow('gui')
cv2.setMouseCallback('gui', mouse_cb)

while(True):
    cv2.imshow('gui',ui.get_screen())
    pressedKey = cv2.waitKey(60)

    if pressedKey == ord('r'):
        if mode == '1':
            mode = '2'
        elif mode =='2':
            mode = '3'
        elif mode == '3':
            mode ='1'
        print mode

    if pressedKey & 0xFF == 27:
        break


cv2.destroyAllWindows()

