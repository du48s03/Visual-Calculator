import cv2
import numpy as np

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
        self.canvas = np.ones((self.size[0],self.size[1], 3L), dtype=np.int8)*255
        self.cursur = Cursor(self.size[0]/2, self.size[1]/2)
        self.color = palette["BLACK"]
        self.bgcolor = palette["WHITE"]

        self.screen = self.get_screen()


    def conv_coord(self, input_coord):
        """Convert the coordinate from the input to the output"""
        #TODO: convter the coordinate!!
        pass

    def drawpoint(self, location):
        cood = self.conv_coord(location)
        cv2.circle(self.canvas, cood, radius=5, color=self.color, thickness=-1)

    def erase(self, location):
        cood = self.conv_coord(location)
        cv2.circle(self.canvas, cood, radius=5, color=self.bgcolor, thickness=-1)

    def setcolor(self, color_name):
        """color_name is a string"""
        self.color = palette[color]

    def settool(self, tool_number):
        """Tool is a string indicating which tool to change to """
        self.cursor.tool = toolmap[tool_number]

    def setcursor(self, location):
        self.cursor.location = self.conv_coord(location)

    def update_screen(self):
        """update the secreen with canvas and cursor"""
        self.screen = self.canvas
        ###TODO: Paint the cursor here!!
        pass

    def get_screen(self):
        """update the screen with canvas and cursor, then return the screen"""
        self.update_screen()
        return self.screen

class Cursor(object):
    """docstring for Cursor"""
    def __init__(self, init_i, init_j):
        self.location = (init_i, init_j)
        self.tool = toolmap[1]
        