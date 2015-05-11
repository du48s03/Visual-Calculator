import cv2
import numpy as np
import pdb
import time
import math
import os
import csv
def drawing(im):
    rgb = cv2.split(im)
    # use only B (because we use Red color to write)
    im2 = rgb[0]
    im2[rgb[0] == 0] = 50
    im2[rgb[0] == 255] = 0
    return im2

def evalcircle(im,center,r):
    im2 = drawing(im)
    (x,y) = im2.nonzero()
    sump = 0.0
    for i in range(len(x)):
        sump += math.fabs(((x[i]-center[0])**2 + (y[i]-center[1])**2)**(0.5) - r)
    sump /= len(x)
    return sump

def evalline(im,start,end):
    # this evaluation cannot be used for lines parallel to x axis
    im2 = drawing(im)
    (x,y) = im2.nonzero()
    min_y = min(y)
    max_y = max(y)
    print min_y,max_y
    sump = 0
    for i in range(20):
        point = [0,0]
        epoint = [0,0]
        point[0] = start[0] + i * (end[0] - start[0]) / 20
        point[1] = start[1] + i * (end[1] - start[1]) / 20
        epoint[0] = min_y + i * (max_y - min_y)/20
        epoint[1] = np.where(im2[:,epoint[0]])[0][0]
        print point, epoint
        sump += evalpoint(epoint,point)
    sump /= 20
    return sump

def evalpoint(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def evalpoints(im,points):
    im2 = drawing(im)
    rows,cols = im2.shape
    (x,y) = im2.nonzero()
    sump = 0
    print points
    for i in range(len(x)):
        print x[i],y[i]
        if x[i] < rows/2 and y[i] < cols/2:
            sump += evalpoint([x[i],y[i]],points[0])
        elif x[i] < rows/2 and y[i] >= cols/2:
            sump += evalpoint([x[i],y[i]],points[1])
        elif y[i] < cols/2:
            sump += evalpoint([x[i],y[i]],points[2])
        else:
            sump += evalpoint([x[i],y[i]],points[3])
    return sump/len(x)

def exp_circle(username):
    if(not os.path.exists('experiments')):
        os.mkdir('experiments')
    f  = open('experiments/circle.csv','ab')
    csvWriter = csv.writer(f)
    # do experiment
    center = [50,90]
    r = 20
    for i in range(5):
        # im,time = exp(center,r)
        ## for debuging
        im = cv2.imread('test.png')
        cv2.circle(im,(90,50),10,(0,0,255),1)
        time = 3
        ## for debugging

        c = evalcircle(im,center,r)
        csvWriter.writerow([username,i,time,c])
        print 'wirte'

def exp_line(username):
    if(not os.path.exists('experiments')):
        os.mkdir('experiments')
    f  = open('experiments/line.csv','ab')
    csvWriter = csv.writer(f)
    # do experiment
    start = (10,10)
    end = (100,100)
    for i in range(1):
        # im,time = exp(center,r)
        ## for debuging
        im = cv2.imread('test.png')
        cv2.line(im,(10,10),(100,110),(0,0,255),1)
        time = 3
        ## for debugging
        c = evalline(im,start,end)
        csvWriter.writerow([username,i,time,c])
        print 'wirte'

def exp_points(username):
    if(not os.path.exists('experiments')):
        os.mkdir('experiments')
    f  = open('experiments/points.csv','ab')
    csvWriter = csv.writer(f)
    # do experiment
    points = [(40,40),(40,350),(350,40),(350,350)]
    for i in range(1):
        # im,time = exp(center,r)
        ## for debuging
        im = cv2.imread('test.png')
        cv2.circle(im,(40,40),1,(0,0,255),1)
        cv2.circle(im,(350,350),1,(0,0,255),1)
        cv2.circle(im,(40,350),1,(0,0,255),1)
        cv2.circle(im,(350,40),1,(0,0,255),1)
        time = 3
        ## for debugging
        c = evalpoints(im,points)
        csvWriter.writerow([username,i,time,c])
        print 'wirte'

if __name__ == '__main__':
    exp_points('aya')
