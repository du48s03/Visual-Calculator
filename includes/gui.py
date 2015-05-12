import cv2
import numpy as np
import posture

#BGR
palette = {
    "WHITE":(255,255,255), 
    "BLACK":(0,0,0), 
    "RED":(0,0,255), 
    "BLUE":(255,0,0),
    "GREEN":(0,255,0),
}

toolmap = {
    1:'pen',
    2:'drag',
    3:'eraser',
}


class GUI(object):
    """docstring for GUI"""
    def __init__(self):
        self.size = (480,640)
        self.canvas = np.ones((480,640, 3L),)*255
        self.cursor = Cursor(self.size[0]/2, self.size[1]/2)
        self.color = palette["BLACK"]
        self.bgcolor = palette["WHITE"]

        self.drawing = False
        self.erasing = False

        self.screen = np.copy(self.canvas)
        self.update_screen()


    def conv_coord(self, input_coord):
        """Convert the coordinate from the input to the output"""
        i,j = input_coord
        center1 = (261.5,322)
        center2 = (self.size[0]/2.0, self.size[1]/2.0)
        di1 = i-center1[0]
        dj1 = j-center1[1]

        scalei = center2[0]*2/222.0
        scalej = center2[1]*2/151.0

        di2 = - di1*scalei #Up side down
        dj2 = - dj1*scalej 

        return (int(center2[0]+di2), int(center2[1]+dj2))

    def drawline(self, loc1, loc2):
        cv2.line(self.canvas, (loc1[1], loc1[0]), (loc2[1], loc2[0]), color=self.color, thickness=5)

    def eraseline(self, loc1, loc2):
        cv2.line(self.canvas, (loc1[1], loc1[0]), (loc2[1], loc2[0]), color=self.bgcolor, thickness=5)

    def setcolor(self, color_name):
        """color_name is a string"""
        self.color = palette[color]

    def settool(self, tool_number):
        """Tool is a string indicating which tool to change to """
        self.cursor.tool = toolmap[tool_number]

    def setcursor(self, location):
        self.cursor.location = location

    def update_screen(self):
        """update the secreen with canvas and cursor"""
        self.screen = np.copy(self.canvas)
        cursor_loc_i = self.cursor.location[0]
        cursor_loc_j = self.cursor.location[1]
        cv2.line(self.screen, (cursor_loc_i,cursor_loc_j-5), (cursor_loc_i,cursor_loc_j+5), (0,0,0), 2)
        cv2.line(self.screen, (cursor_loc_i-5,cursor_loc_j), (cursor_loc_i+5,cursor_loc_j), (0,0,0), 2)

    def get_screen(self):
        """update the screen with canvas and cursor, then return the screen"""
        self.update_screen()
        return self.screen

    def handle_input(self, label, location, isTouching):
        """The method to handle the signals from the device"""
        cvt_coord = self.conv_coord(location)
        #paint first
        if self.drawing and label == posture.poses['POINTING'] and isTouching:
            self.drawline(self.cursor.location, cvt_coord)
        if self.erasing and label == posture.poses['PALM'] and isTouching:
            self.eraseline(self.cursor.location, cvt_coord)

        #Color selection
        # if not self.drawing and isTouching and label == posture.poses['POINTING']:#Finger down event

        # if not self.erasing and isTouching and label == posture.poses['PALM']:#Finger down event        


        #Finally, update state
        self.drawing  = label == posture.poses['POINTING'] and isTouching
        self.erasing  = label == posture.poses['PALM'] and isTouching
        self.setcursor(cvt_coord)

        self.update_screen()

    def save_canvas(self,filename):
        """ save canvas """
        print 'save images'
        cv2.imwrite(filename,self.canvas)

    def draw_sample(self,image):
        """ draw sample image on canvas """
        self.canvas = image

    def handle_input_m(self, label, location, isTouching):
        """The method to handle the signals from the mouse"""
        cvt_coord = location
        #paint first
        if self.drawing and label == posture.poses['POINTING'] and isTouching:
            self.drawline(self.cursor.location, cvt_coord)
        if self.erasing and label == posture.poses['PALM'] and isTouching:
            self.eraseline(self.cursor.location, cvt_coord)
        #Finally, update state
        self.drawing  = label == posture.poses['POINTING'] and isTouching
        self.erasing  = label == posture.poses['PALM'] and isTouching
        self.setcursor_m(cvt_coord)

        self.update_screen()

    def setcursor_m(self, location):
        self.cursor.location = location


class Cursor(object):
    """docstring for Cursor"""
    def __init__(self, init_i, init_j):
        self.location = (init_i, init_j)
        self.tool = toolmap[1]
        
