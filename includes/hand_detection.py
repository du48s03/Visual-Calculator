import cv2
import numpy as np
import pdb
import time
"""Package for hand detection"""
def unique(a):
    """ remove duplicate columns and rows
        from http://stackoverflow.com/questions/8560440 """
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

def skin_detection(image, haarcascades, flood_diff=7, min_face_size=(6,6),
        num_iter=5, step=3):
    # face detection
    faces = haarcascades.detectMultiScale(image, minSize=min_face_size)
    mask = np.zeros(image.shape)
    hsv_im = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    if len(faces) >= 0:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                r = image[y,x,2]
                g = image[y,x,1]
                b = image[y,x,0]
                s = hsv_im[y,x,1]
                v = hsv_im[y,x,2]
                if r > 95 and g > 40 and b > 20 and max(r,g,b) - min(r,g,b) > 15 and abs(r-g) > 15 and r > g and r > b and s > 50:
                        mask[y,x,:] = 255
        if len(faces) > 0:
            mask[faces[0][1]:faces[0][1]+faces[0][3],faces[0][0]:faces[0][0]+faces[0][2]] = 0
    if len(faces) > 0:
        mask2 = mask
        # convert to hsv
        image_original = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image2 = np.copy(image_original)
        image2[image2==[0,0,0]] = 1

        skin_color = np.zeros((len(faces), 3))
        for i, face in enumerate(faces):
            image_face = image2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
            skin_color[i] = np.array([image_face[image_face.shape[0]/2, image_face.shape[1]/2]])
        mask = np.zeros(image_original.shape)
        #### osoi!!!!
        for i in range(num_iter):
        # for each pixel, call floodFill(image2) if it is close to skin_color
            for y in range(0, image2.shape[0], step):
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
        mask2[face[1]:face[1]+face[3],face[0]:face[0]+face[2]] = 0
        if np.sum(mask) < 20000:
            print 'tisai'
            mask = mask2
    k = np.ones((3,3),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.resize(mask,(mask.shape[1]*4,mask.shape[0]*4))
    mask = np.bool_(mask)
    return mask
def grab(im):
    im_s = cv2.resize(im,(im.shape[1]/4,im.shape[0]/4))
    # rect
    x2 = im_s.shape[1]*19/20
    y2 = im_s.shape[0]#*4/5
    x1 = 0 #im_s.shape[1]*1/40
    y1 = im_s.shape[0]*5/16
    # masks
    mask = np.zeros(im_s.shape[:2],np.uint8)
    bgd_model = np.zeros((1,65),np.float64)
    fgd_model = np.zeros((1,65),np.float64)
    rect = (x1,y1,x2,y2)
    # grabcut
    cv2.grabCut(im_s,mask,rect,bgd_model,fgd_model,3,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    #mask2 = cv2.resize(mask2,(mask2.shape[1]*8,mask2.shape[0]*8))
    im_s = im_s*mask2[:,:,np.newaxis]
    return im_s
def hand_detection(im):
    #filename = "testdata/Image" + ("00"+str(i))[-2:] + ".bmp"
    #print filename
    #filename_out = (''.join(filename.split('.')[:-1])
    #                + '_out_1_%d.' + filename.split('.')[-1])
    # for face detection
    hc = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    # load image
    #im = cv2.imread(filename)
    im2 = grab(im)
    #cv2.imwrite(filename_out % 4, im2)
    # Skin detection using face information
    mask = skin_detection(im2, hc)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(gray,127,255,0)[1]
    th = np.bool_(th)
    # pdb.set_trace()
    #cv2.imwrite(filename_out % 1, im)
    #cv2.imwrite(filename_out % 2, np.uint8(mask)*255)
    #im[mask==False] = 0
    #cv2.imwrite(filename_out % 3, im)
    return th
