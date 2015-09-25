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
import scipy.ndimage.morphology
import argparse
import timeout_decorator

logging.basicConfig(stream = sys.stderr, level=logging.INFO)
# 1. get kinect input
# 2. bounding box calculation

parser = argparse.ArgumentParser(description='Track Motion!')
parser.add_argument('-p', dest='motion', action='store_true',help="Start in motion capture mode")
parser.add_argument('-osc', dest='osc', default=7770,help="OSC Port")
parser.set_defaults(motion=False,osc=7770)
args = parser.parse_args()

target = liblo.Address(7770)
def send_osc(path,*args):
    global target
    return liblo.send(target, path, *args)


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
    return depth.astype(np.int)

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
    ''' returns z,x,y '''
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

curr_state = None

handlers = dict()

def handle_keys():
    global fullscreen
    global handlers
    global curr_state
    k = cv2.waitKey(1000/60) & 0xff
    if k == 27:
        return True
    elif k == ord('p'):
        logging.info('Switching to transition state')
        curr_state = TransitionState()
    elif k == ord('s'):
        logging.info('Switching to stillness state')
        curr_state = Stillness()
    elif k == ord('f'):
        doFullscreen()
    else:
        if k in handlers:
            handlers[k]()
    return False

class Bounder(object):
    def __init__(self):
        self.cb = None
    def matches(self,depthMap,xMat,yMat,zMat):
        return False
    def cb(self,**kwargs):
        if not self.cb == None:
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

kernel = np.ones((5,5))



class SphereBounder(Bounder):
    def __init__(self,center,radius):
        super(SphereBounder,self).__init__()
        self.center = center
        self.radius = radius
        self.mini   = 1024*np.ones((WIDTH,HEIGHT))
        self.maxi   = 0*np.zeros((WIDTH,HEIGHT))
        self.callback = None

    def matches(self,depthMap,xMat,yMat,zMat):
        center = self.center
        #(cz,cx,cy) = point_to_depth_map(center[0],center[1],center[2])
        (cx,cy,cz) = center
        rad = self.radius ** 3
        equation = (xMat - float(cx)) ** 2 + (yMat - float(cy)) ** 2 + (zMat - float(cz)) ** 2
        matches = equation < (self.radius ** 3)
        cv2.imshow("equation",equation)

        match_sum = np.sum(matches)
        logging.info("SPhere match sum %s %s" % (match_sum, rad))
        return match_sum > 100 # choose something better

class Context(object):
    def __init__(self):
        self.diff = None
        self.lastframe = None
        self.diffsum = 0
        self.seen = 0
        self.mask = None

    def get_centroid(self):
        return (self.cx,self.cy,self.cz)

    def add_frame(self,depth,xMat,yMat,zMat):
        depth = depth.astype(np.int)
        if self.seen == 0:
            self.diff = depth * 0
            self.lastframe = depth
            self.diffsum = 0
        if self.mask == None:
            self.mask = self.depth >= 0
        self.diff = np.abs((self.mask & (depth > 0) & (self.lastframe > 0)) * (depth - self.lastframe))
        #self.diff = np.abs((self.mask & (depth > 0) ) * (depth - self.lastframe))
        # self.diff = self.diff**2

        #self.diff = scipy.ndimage.morphology.grey_erosion(self.diff,size=(3,3))

        #self.diff = np.abs(depth - self.lastframe)
        #self.diff = (self.diff < 10) * self.diff
        self.lastframe = depth
        self.diffsum = sum(self.diff)
        booldiff = self.diff > 0
        boolcount = np.sum(booldiff)
        logging.debug("boolcount %s" % boolcount)
        # fix average to existing points
        self.cx = np.sum(booldiff * xMat)/float(boolcount)
        self.cy = np.sum(booldiff * yMat)/float(boolcount)
        self.cz = np.sum(booldiff * zMat)/float(boolcount)
        logging.debug("[%s,%s] %s %s %s [%s,%s,%s]" % (np.max(self.lastframe),np.max(depth),self.diffsum/float(640*480),np.min(self.diff),np.max(self.diff),self.cx, self.cy, self.cz))
        self.seen += 1

    def detect_stillness(self, threshold=100):
        thresh = (WIDTH * HEIGHT * 0.5)
        logging.debug("%s < %s" % (self.diffsum,thresh))
        return self.diffsum < thresh 

class Commander(object):
    def __init__(self):
        self.queue = list()
    def add(self, command):
        logging.info("Adding command: %s " % command)
        self.queue.append(command)
    def execute_queue(self):
        l = self.queue
        self.queue = list()
        for command in l:
            logging.info("Executing %s" % command)
            command.execute(self)

commander = Commander()
positions = list()
bounds = list()

def addPosition(centroid, radius=1.1):
    global positions
    global bounds
    positions.append(centroid)
    posi = len(positions) + 1
    bounder = SphereBounder(centroid, radius)
    def responder():
        logging.warn("Centroid: (%s,%s,%s) reached!" % centroid)
        send_osc("/kinect/position_trigger",int(posi),
                 float(centroid[0]),
                 float(centroid[1]),
                 float(centroid[2]))
    bounder.cb = responder
    bounds.append(bounder)
    



class Command(object):
    def __init(self):
        '''nothing'''
    def execute(self,commander):
        '''nothing'''
        
# 0. start
# 1. detect motion
# 1.1 find centroid
# 1.2 remember centroid
# 2. detect stillness
# 2.1 count stillness time
# 3. detect motion
# 3.1 find centroid
# 3.2 count for activation

# def detect_stillness(diffmat, threshold=100):
#     stillness = np.sum( )
#     logging.warn("Stillness %s < %s" % (stillness,threshold))
#     return stillness <  threshold


class State(object):
    def __init__(self):
        self._nextstate = self
    def transition(self, state):
        self._nextstate = state
    def nextstate(self):
        return self._nextstate

class Stillness(State):
    def __init__(self):
        super(Stillness,self).__init__()
        self.steps = 0
        self.threshold = 100

    def step(self,context):
        logging.info("State:Stillness")
        self.steps += 1
        if context.detect_stillness():
            self.transition(self)
        else:
            self.steps = 0
            self.transition(InMotion())

default_state = Stillness()

class InMotion(State):
    def __init__(self):
        super(InMotion,self).__init__()
        self.steps = 0
    def step(self,context):
        logging.info("State:InMotion")
        self.steps += 1
        if not context.detect_stillness():
            self.transition(self)
        else:
            self.steps = 0
            self.transition(Stillness())

class TransitionState(State):
    def __init__(self):
        super(TransitionState,self).__init__()
        self.steps = 0
        self.stills = 0
        self.motions = 0
        self.stillthreshold = 10
    def step(self,context):
        logging.info("State:TransitionState")
        self.steps += 1
        self.steps += 1
        stillnow = context.detect_stillness()
        if stillnow:
            self.stills += 1
        else:
            self.motions += 1
        if self.stills > self.stillthreshold and not stillnow:
            self.transition(MoveToPosition())
        else:
            self.transition(self)
        

class MoveToPosition(State):    
    ''' Basically wait till motion stops for a second or so '''
    def __init__(self):
        super(MoveToPosition,self).__init__()
        self.steps = 0
        self.stills = 0
        self.motions = 0
        self.stillthreshold = 30
    def step(self,context):
        logging.info("State:MoveToPosition ")
        self.steps += 1
        stillnow = context.detect_stillness()
        if stillnow:
            self.stills += 1
        else:
            self.motions += 1
            self.centroid = context.get_centroid()
        
        # wait till motion!
        if self.stills > self.stillthreshold and not stillnow: 
            self.transition(SetPosition(centroid = self.centroid))
        elif self.stills >  6*self.stillthreshold and stillnow:
            logging.info("Uh nothing is going on")
            self.transition(TransitionState())
        else:
            self.transition(self)


class NewPositionCommand(Command):
    def __init__(self,centroid):
        super(NewPositionCommand,self).__init__()
        self.centroid = centroid
    def execute(self,commander):
        '''I don't know!'''
        logging.info("NewPositionCommand: (%s,%s,%s)" % self.centroid)
        addPosition(self.centroid)

def mix_point(c1, c2, prop=0.5):
    dprop = 1.0 - prop
    return (c1[0]*prop + c2[0]*dprop,
            c1[1]*prop + c2[1]*dprop,
            c1[2]*prop + c2[2]*dprop)

def min_max_point(minmax, newpoint):
    return (
        (
            max(minmax[0][0],newpoint[0]),
            max(minmax[0][1],newpoint[1]),
            max(minmax[0][2],newpoint[2])
        ),
        (
            min(minmax[1][0],newpoint[0]),
            min(minmax[1][1],newpoint[1]),
            min(minmax[1][2],newpoint[2])
        )
    )

def center_min_max(minmax):
    return ((minmax[0][0]+minmax[1][0])/2.0,
            (minmax[0][1]+minmax[1][1])/2.0,
            (minmax[0][2]+minmax[1][2])/2.0)


class SetPosition(State):    
    ''' We start in motion and wait for stillness'''
    def __init__(self,centroid):
        super(SetPosition,self).__init__()
        self.steps = 0
        self.stills = 0
        self.constill = 0        
        self.motions = 0
        self.motion_threshold = 15
        self.centroid = centroid
        self.minmax = min_max_point((centroid,centroid),centroid)
        self.original_centroid = centroid
    def step(self,context):
        logging.info("State:SetPosition %s %s" % (self.stills, self.motions))
        self.steps += 1
        stillnow = context.detect_stillness()
        if stillnow:
            self.stills += 1
            self.constill += 1
        else:
            self.motions += 1
            self.constill = 0
            self.centroid = mix_point(self.centroid,context.get_centroid(),0.5)
            self.minmax = min_max_point(self.minmax,self.centroid)

        if self.motions > self.motion_threshold and self.constill > self.motion_threshold and stillnow:             
            pt = center_min_max(self.minmax)
            pt = (pt[0],pt[1],pt[2])
            commander.add(NewPositionCommand(pt))
            self.transition(default_state)
        else:
            self.transition(self)

            
        
def communicate_centroid(centroid):
    logging.info("Centroid sending (%s,%s,%s)" % centroid)
    send_osc("/kinect/centroid",float(centroid[0]),float(centroid[1]),float(centroid[2]))
    
def communicate_positions(positions):
    for i in range(0,len(positions)):
        pos = positions[i]
        logging.info("Pos sending %s (%s)" % (i,pos))
        send_osc("/kinect/position",int(i),float(pos[0]),float(pos[1]),float(pos[2]))


                


starttime = None
context = Context()
curr_state = Stillness()
if args.motion:
    curr_state = TransitionState()



# step 1 get noise map

def get_mask(n=15):

    summap = get_depth_map().astype(np.int)
    summap *= 0
    n = 15
    for i in range(0,n):
        depth_map = get_depth_map().astype(np.int)
        depth_map = scipy.ndimage.morphology.grey_erosion(depth_map,size=(3,3))

        summap += (depth_map < 1) | (depth_map > 1020)
        logging.debug("%s %s" % (np.min(depth_map), np.max(depth_map)))
        logging.debug("%s %s %s" % (np.min(summap),np.max(summap),(np.sum(summap)/float(640*480))))

    mask = (summap == n) | (summap < (n*1/20)) * 1
    return (mask, summap)

context.mask, summap = get_mask(15)
#cv2.imshow("summap",200*summap/60.0)
#cv2.imshow("mask",np.ones(context.mask.shape)*context.mask)

# UI plans
# press p to set state to transition for setting a position
# press s to reset to stillness
# - [ ] need to add positional triggers    

def check_bounds(depthMap,xMat,yMat,zMat,bound_list = None):
    if bound_list == None:
        bound_list = bounds
    for bounder in bound_list:
        bounder.match_w_cb(depthMap,xMat,yMat,zMat)

while(1):
    depth_map = get_depth_map()
    if depth_map == None:
        print "Bad?"
        continue
    x,y,z = depth_map_to_points(depth_map)
    context.add_frame(depth_map,x,y,z)
    curr_state.step(context)
    curr_state = curr_state.nextstate()

    commander.execute_queue()

    check_bounds(depth_map,x,y,z,bound_list=bounds)

    # now communicate the centroid if there is motion
    
    if not context.detect_stillness():
        communicate_centroid(context.get_centroid())
        communicate_positions(positions)

    depth_map_bmp = depth_map_to_bmp(depth_map)
    depth_map_bmp = cv2.flip(depth_map_bmp, 1)
    cv2.imshow(screen_name,(255/8) * (depth_map_bmp % 8))
    # cv2.imshow("%s - diff" % screen_name,depth_map_to_bmp(context.diff))# * context.diff))
    cv2.imshow("%s - diff" % screen_name,(context.diff == 0)*255.0)# * context.diff))

    if handle_keys():
        break


cap.release()
cv2.destroyAllWindows()
