# coding=utf8
import sys
import cv2
import numpy as np
from numpy import *
import freenect
import liblo
import random
import time
import logging

logging.basicConfig(stream = sys.stderr, level=logging.INFO)
# 1. get kinect input
# 2. bounding box calculation

screen_name = "KinectMixer"

def current_time():
    return int(round(time.time() * 1000))

target = liblo.Address(57120)

fullscreen = False
cv2.namedWindow(screen_name, cv2.WND_PROP_FULLSCREEN)

kinect = None
cap = None
    
"""
Grabs a depth map from the Kinect sensor and creates an image from it.
http://euanfreeman.co.uk/openkinect-python-and-opencv/
"""
def get_depth_map():    
    depth, timestamp = freenect.sync_get_depth()
    np.clip(depth, 0, 2**10 - 1, depth)
    return depth

def depth_map_to_bmp(depth):
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

WIDTH=640
HEIGHT=480
Y,X = np.meshgrid(np.arange(0,HEIGHT),np.arange(0,WIDTH),indexing='ij')
kinectXmat = X - (WIDTH/2.0)
kinectYmat = Y - (HEIGHT/2.0)
Min_Dist = -10.0
Scale_Factor = 0.0021

dmfloor,dmwall = np.meshgrid(np.arange(0,HEIGHT),np.arange(0,WIDTH),indexing='ij')
dmfloor = 1024 - 1024 * dmfloor / 480

"""
	Conversion from http://openkinect.org/wiki/Imaging_Information
"""
def depth_map_to_points(dm):
    z = 0.1236 * np.tan(dm / 2842.5 + 1.1863)
    x = kinectXmat * (z + Min_Dist) * Scale_Factor
    y = kinectYmat * (z + Min_Dist) * Scale_Factor
    #x = (i - w / 2) * (z + minDistance) * scale_factor
    #y = (j - h / 2) * (z + minDistance) * scale_factor
    #z = z
    #Where
    #minDistance = -10
    #scaleFactor = .0021.
    return (x,y,z)

def points_to_depth_map(x,y,z):
    dm = 2842.5 * (np.arctan(z / 0.1236)  - 1.1863)
    ourX = (x / Scale_Factor) / (z + Min_Dist) 
    ourY = (y / Scale_Factor) / (z + Min_Dist) 
    return (dm,ourX,ourY)

def point_to_depth_map(x,y,z):
    return points_to_depth_map(x,y,z) # ha it works
    

def get_kinect_video():    
    if not kinect == None:
        return get_kinect_video_cv()
    depth, timestamp = freenect.sync_get_video()  
    if (depth == None):
        return None
    return depth[...,1]



def get_kinect_video_cv():    
    global kinect
    if kinect == None:
        print "Opening Kinect"
        kinect = cv2.VideoCapture(1)
    ret, frame2 = kinect.read()
    if not ret:
        return None
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    return next




def doFullscreen():
    global fullscreen
    if not fullscreen:
        cv2.setWindowProperty(screen_name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        fullscreen = True
    else:
        cv2.setWindowProperty(screen_name, cv2.WND_PROP_FULLSCREEN, 0)
        fullscreen = False

handlers = dict()

def handle_keys():
    global fullscreen
    global handlers
    k = cv2.waitKey(1000/60) & 0xff
    if k == 27:
        return True
    #elif k == ord('s'):
    #    #cv2.imwrite('opticalfb.png',frame2)
    #    #cv2.imwrite('opticalhsv.png',rgb)
    elif k == ord('f'):
        doFullscreen()
    else:
        if k in handlers:
            handlers[k]()
    return False

class Bounder(object):
    def __init__(self):
        super(self)
        self.cb = None
    def matches(self,depthMap,xMat,yMat,zMat):
        return False
    def cb(self,**kwargs):
        if not cb == None:
            self.cb(self,**kwargs)
    def match_w_cb(self,depthMap,xMat,yMat,zMat):
        if (self.matches(depthMap,xMat,yMat,zMat)):
            self.cb()
            
class BoxBounder(Bounder):
    def __init__(self,top,bottom):
        super(self)
        """ assume top is front """
        self.top = top
        self.bottom = bottom

    def matches(self,depthMap,xMat,yMat,zMat):
        top = self.top
        bottom = self.bottom
        matches = (xMat >= top[0] ) & (xMat <= bottom[0]) & (yMat >= top[1]) & (yMat <= bottom[1]) & ( zMat >= top[2]) & (zMat <= bottom[2])
        return np.sum(matches) > 100


class SphereBounder(Bounder):
    def __init__(self,center,radius):
        super(self)
        self.center = center
        self.radius = radius
        self.mini   = 1024*np.ones((WIDTH,HEIGHT))
        self.maxi   = 0*np.zeros((WIDTH,HEIGHT))
        self.callback = None
        #self.render_sphere()

    # def render_sphere(self):
    #     center = self.center
    #     radius = self.radius
    #     (cz,cx,cy) = point_to_depth_map(center[0],center[1],center[2])
    #     (czf,_,_) = point_to_depth_map(center[0],center[1],center[2]+radius)
    #     self.drad = np.abs(czf - cz)
    #     (clz,clx,cly) = point_to_depth_map(center[0]-radius,center[1],center[2])
    #     prad = clx - cx
    #     xx, yy = numpy.mgrid[WIDTH, HEIGHT]
    #     circle = (kinectXmat - cx) ** 2 + (kinectYmat - cy) ** 2 < prad*prad
    #     self.circle = circle*1

    def matches(self,depthMap,xMat,yMat,zMat):
        center = self.center
        (cz,cx,cy) = point_to_depth_map(center[0],center[1],center[2])
        matches = (xMat - cx) ** 2 + (yMat - cy) ** 2 + (zMat - cz) ** 2 < (radius ** 3)
        return np.sum(matches) > 100 # choose something better

class Context(object):
    def __init__(self):
        self.diff = None
        self.lastframe = None
        self.diffsum = 0
        self.seen = 0

    def add_frame(self,depth,xMat,yMat,zMat):
        if self.seen == 0:
            self.lastdiff = depth * 0.0
            self.diff = depth * 0.0
            self.lastframe = depth
            self.diffsum = 0
        self.lastdiff = self.diff
        self.diff = depth - self.lastframe
        self.lastframe = depth
        self.diffsum = sum(self.diff)
        booldiff = diff > 0
        self.cx = np.average(booldiff * xMat)
        self.cy = np.average(booldiff * yMat)
        self.cz = np.average(booldiff * zMat)
        self.seen += 1
        
        
# 0. start
# 1. detect motion
# 1.1 find centroid
# 1.2 remember centroid
# 2. detect stillness
# 2.1 count stillness time
# 3. detect motion
# 3.1 find centroid
# 3.2 count for activation

def detect_stillness(diffmat, threshold=100):
    return np.sum(( diffmat > 0 ) & (diffmat < 1000) * diffmat) < threshold

class State(object):
    def __init__(self):
        self._nextstate = self
    def transition(self, state):
        self._nextstate = self
    def nextstate(self):
        return self._nextstate

class InMotion(State):
    def __init__(self):
        super(InMotion,self).__init__()
        self.steps = 0
    def step(self,context):
        logging.info("State:InMotion")
        self.steps += 1
        if not still(context.diff):
            self.transition(self)
        else:
            self.steps = 0
            self.transition(Stillness())

    

class Stillness(State):
    def __init__(self):
        super(Stillness,self).__init__()
        self.steps = 0
        self.threshold = 100

    def step(self,context):
        logging.info("State:Stillness")
        self.steps += 1
        if still(context.diff,self.threshold):
            self.transition(self)
        else:
            self.steps = 0
            self.transition(InMotion())
                

starttime = None
context = Context()
curr_state = Stillness()
while(1):

    depth_map = get_depth_map()
    if depth_map == None:
        print "Bad?"
        continue
    context.add_frame(depth_map)
    curr_state.step(context)
    curr_state = curr_state.nextstate()

    depth_map_bmp = depth_map_to_bmp(depth_map)
    depth_map_bmp = cv2.flip(depth_map_bmp, 1)
    cv2.imshow(screen_name,depth_map_bmp)

    if handle_keys():
        break


cap.release()
cv2.destroyAllWindows()
