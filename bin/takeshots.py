import cv2

cap = cv2.VideoCapture(1)

index = 0
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    pressedkey = cv2.waitKey(20)
    if pressedkey == ord('q'):
        break
    elif pressedkey == ord('z'): 
        #take one shot
        cv2.imwrite('testimage'+str(index)+'.png', frame)
        index+=1