import cv2
import numpy as np
import pdb

"""Package for hand detection"""
def hand_detection(im):
   hc = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
    x2 = im.shape[0] - 1
    y2 = im.shape[1]* 4 / 5
    x1 = 0
    y1 = im.shape[1]* 1 / 5
    mask = np.zeros(im.shape[:2],np.uint8)
    bgd_model = np.zeros((1,65),np.float64)
    fgd_model = np.zeros((1,65),np.float64)
    rect = (x1,y1,x2,y2)
    cv2.grabCut(im,mask,rect,bgd_model,fgd_model,2,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    im = im*mask2[:,:,np.newaxis]
    mask = skin_detection(im, hc, verbose=False)
    return mask

def unique(a):
    """ remove duplicate columns and rows
        from http://stackoverflow.com/questions/8560440 """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

def skin_detection(image, haarcascades, flood_diff=4, min_face_size=(30,30),
        num_iter=3, verbose=False, step=1):
    faces = haarcascades.detectMultiScale(image, minSize=min_face_size)
    if len(faces) == 0:
        raise Exception('no faces')

    image_original = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image2 = np.copy(image_original)
    image2[image2==[0,0,0]] = 1

    skin_color = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        image_face = image2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
        skin_color[i] = np.array([image_face[image_face.shape[0]/2, image_face.shape[1]/2]])

    mask = np.zeros(image_original.shape)
    for i in range(num_iter):
        # for each pixel, call floodFill(image2) if it is close to skin_color
        for y in range(0, image2.shape[0], step):
            if verbose:
                print 'iter: %d, y:%d' % (i, y)
            for x in range(0, image2.shape[1], step):
                color = image2[y,x]
                if (color!=(0,0,0)).any():
                    if any((np.abs(skin_color-color)<=(flood_diff,)*3).all(1)):
                        cv2.floodFill(image2, None, (x,y), (0, 0, 0),
                                loDiff=(flood_diff,)*3, upDiff=(flood_diff,)*3)

        # update mask image and skin_color
        mask[image2==(0,0,0)] = 255
        skin_color = image_original[mask.nonzero()]
        skin_color.shape = (skin_color.shape[0]/3, 3)
        skin_color = unique(skin_color)
    mask[face[1]:face[1]+face[3],face[0]:face[0]+face[2]] = 0
    mask = np.bool_(mask)
    return mask
