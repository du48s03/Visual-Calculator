import cv2
import numpy as np
import pdb
import time
import math
import os
import csv
import eval_ui as eu
import mousepaint as mo
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
    x = np.sort(x)
    min_y = min(y)
    max_y = max(y)
    
    print min_y,max_y
    sump = 0
    for i in range(20):
        point = [0,0]
        epoint = [0,0]
        point[0] = start[0] + i * (end[0] - start[0]) / 20
        point[1] = start[1] + i * (end[1] - start[1]) / 20
        epoint[0] = x[int(len(x) * i / 20)]
        ys = np.where(im2[epoint[0],:])[0]
        if (len(ys) == 0):
            raise IndexError("Out of bound")
        else:
            epoint[1] = ys[0]
        #print point, epoint
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
    #print points
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
    if(not os.path.exists('../experiments')):
        os.mkdir('../experiments')
    f  = open('../experiments/circle.csv','ab')
    csvWriter = csv.writer(f)
    ff = open('../experiments/std.csv','ab')
    csvWriter2 = csv.writer(ff)
    # do experiment
    center = [250,290]
    r = 190
    ts = []
    cs = []
    for i in range(3):
        ## for debuging
        im2 = cv2.imread('../test.png')
        cv2.circle(im2,(290,250),190,(255,0,0),1)
        filename = '../experiments/circle_'+username+'_'+str(i) + '.png'
        im,time = eu.evaluation(filename,im2)
        c = evalcircle(im,center,r)
        csvWriter.writerow([username,'VI', i , round(time,3),round(c,3)])
        ts.append(time)
        cs.append(c)
        if i == 2:
            csvWriter2.writerow(['circle',username,'VI',round(np.std(ts),3),round(np.std(cs),3)])
        print 'wirte'

def exp_line(username):
    if(not os.path.exists('../experiments')):
        os.mkdir('../experiments')
    f  = open('../experiments/line.csv','ab')
    csvWriter = csv.writer(f)
    ff = open('../experiments/std.csv','ab')
    csvWriter2 = csv.writer(ff)
    ts = []
    cs = []
    # do experiment
    start = (50,50)
    end = (400,400)
    for i in range(3):
        ## for debuging
        im2 = cv2.imread('../test.png')
        cv2.line(im2,(50,50),(400,400),(255,0,0),1)
        filename = '../experiments/line_'+username+'_'+str(i) + '.png'
        im,time = eu.evaluation(filename,im2)
        ## for debugging
        c = evalline(im,start,end)
        csvWriter.writerow([username,'VI', i , round(time,3),round(c,3)])
        ts.append(time)
        cs.append(c)
        if i == 2:
            csvWriter2.writerow(['line',username,'VI',round(np.std(ts),3),round(np.std(cs),3)])
 
def exp_points(username):
    if(not os.path.exists('../experiments')):
        os.mkdir('../experiments')
    f  = open('../experiments/points.csv','ab')
    csvWriter = csv.writer(f)
    ff = open('../experiments/std.csv','ab')
    csvWriter2 = csv.writer(ff)
    # do experiment
    ts = []
    cs = []
    points = [(40,40),(40,350),(350,40),(350,350)]
    for i in range(3):
        im2 = cv2.imread('../test.png')
        cv2.circle(im2,(40,40),3,(255,0,0),1)
        cv2.circle(im2,(350,350),3,(255,0,0),1)
        cv2.circle(im2,(40,350),3,(255,0,0),1)
        cv2.circle(im2,(350,40),3,(255,0,0),1)
        filename = '../experiments/points_'+username+'_'+str(i) + '.png'
        im,time = eu.evaluation(filename,im2)
        c = evalpoints(im,points)
        csvWriter.writerow([username,'VI', i , round(time,3),round(c,3)])
        ts.append(time)
        cs.append(c)
        if i == 2:
            csvWriter2.writerow(['points',username,'VI',round(np.std(ts),3),round(np.std(cs),3)])
 
def exp_circle_m(username):
    if(not os.path.exists('../experiments')):
        os.mkdir('../experiments')
    f  = open('../experiments/circle.csv','ab')
    csvWriter = csv.writer(f)
    ff = open('../experiments/std.csv','ab')
    csvWriter2 = csv.writer(ff)
   # do experiment
    center = [250,290]
    r = 190
    ts = []
    cs = []
    for i in range(3):
        ## for debuging
        im2 = cv2.imread('../test.png')
        cv2.circle(im2,(290,250),190,(255,0,0),1)
        filename = '../experiments/circle_'+username+'_'+str(i) + '_m.png'
        im,time = mo.evaluation(filename,im2)
        c = evalcircle(im,center,r)
        csvWriter.writerow([username,'mouse',i,round(time,3),round(c,3)])
        print 'wirte'
        ts.append(time)
        cs.append(c)
        if i == 2:
            csvWriter2.writerow(['circle',username,'mouse',round(np.std(ts),3),round(np.std(cs),3)])
 
def exp_line_m(username):
    if(not os.path.exists('../experiments')):
        os.mkdir('../experiments')
    f  = open('../experiments/line.csv','ab')
    csvWriter = csv.writer(f)
    ff = open('../experiments/std.csv','ab')
    csvWriter2 = csv.writer(ff)
    ts = []
    cs = []
 
    # do experiment
    start = (50,50)
    end = (400,400)
    for i in range(3):
        ## for debuging
        im2 = cv2.imread('../test.png')
        cv2.line(im2,(50,50),(400,400),(255,0,0),1)
        filename = '../experiments/line_'+username+'_'+str(i) + '_m.png'
        im,time = mo.evaluation(filename,im2)
        ## for debugging
        c = evalline(im,start,end)
        csvWriter.writerow([username,'mouse',i,round(time,3),round(c,3)])
        ts.append(time)
        cs.append(c)
        if i == 2:
            csvWriter2.writerow(['line',username,'mouse',round(np.std(ts),3),round(np.std(cs),3)])
 
def exp_points_m(username):
    if(not os.path.exists('../experiments')):
        os.mkdir('../experiments')
    f  = open('../experiments/points.csv','ab')
    csvWriter = csv.writer(f)
    ff = open('../experiments/std.csv','ab')
    csvWriter2 = csv.writer(ff)
    # do experiment
    points = [(40,40),(40,350),(350,40),(350,350)]
    ts = []
    cs = []
    for i in range(3):
        im2 = cv2.imread('../test.png')
        cv2.circle(im2,(40,40),3,(255,0,0),1)
        cv2.circle(im2,(350,350),3,(255,0,0),1)
        cv2.circle(im2,(40,350),3,(255,0,0),1)
        cv2.circle(im2,(350,40),3,(255,0,0),1)
        filename = '../experiments/points_'+username+'_'+str(i) + '_m.png'
        im,time = mo.evaluation(filename,im2)
        c = evalpoints(im,points)
        csvWriter.writerow([username,'mouse',i,round(time,3),round(c,3)])
        ts.append(time)
        cs.append(c)
        if i == 2:
            csvWriter2.writerow(['points',username,'mouse',round(np.std(ts),3),round(np.std(cs),3)])
 
if __name__ == '__main__':
    exp_points('aya')
