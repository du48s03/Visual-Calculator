import cv2
import sys

def mouse_callback(event,x,y,flags,param):
    i,j = (y-1,x-1)
    global counter
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print x,y

cap = cv2.VideoCapture(1)
cv2.namedWindow('input')
cv2.setMouseCallback('input',mouse_callback)

rec = False
# framename_base = sys.argv[1]
# counter = 0
while(True):
    ret, frame = cap.read()
    cv2.imshow('input', frame)
    pressedKey = cv2.waitKey(60)
    if pressedKey == ord('a'):
        rec = not rec
    # if rec:
    #     framename = framename_base+str(counter)+'.png'
    #     cv2.imwrite(framename, frame)
    #     counter = counter+1

    if pressedKey & 0xFF == 27:
        cv2.imwrite('testimg.png', frame)
        break