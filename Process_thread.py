from threading import Thread
import cv2 as cv
from utils import *
import time

# +---------------------------------------+
# | Colour Recognition
# +---------------------------------------+

class Colour_Rec_thread:
    def __init__(self, currframe, currframe_hsv, green_bounds, blue_bounds):
        self.frame = currframe
        self.frame_hsv = currframe_hsv

        self.green_lower = green_bounds[0]
        self.green_upper = green_bounds[1]

        self.blue_lower = blue_bounds[0]
        self.blue_upper = blue_bounds[1]

        self.points = []
        self.cup = 0

    
    def start(self):
        T = Thread(target = self.colour_recognition)
        T.start()
        return T
    
    def colour_recognition(self):
        p = find_points_all(self.frame, self.frame_hsv, self.green_lower, self.green_upper)
        c = find_points_max(self.frame, self.frame_hsv, self.blue_lower, self.blue_upper)
        self.points = p
        self.cup = c


# +---------------------------------------+
# | Point Continuity and Enqueue
# +---------------------------------------+

class Enqueue_thread:
    def __init__(self, currframe):
        self.frame = currframe
        self.points = []
        self.frame_queues = []

        # Time stuff
        self.ref_time = time.time()
        self.curr_time = time.time()
        self.last_time = self.curr_time
    
    def start(self):
        T = Thread(target = self.continuity_and_enqueue) 
        T.start()
        return T
    
    def continuity_and_enqueue(self):
        # Update time variables
        self.last_time = self.curr_time
        self.curr_time = time.time() - self.ref_time
        dt = self.curr_time - self.last_time

        if len(self.points) > 0: # If a ball has been detected
            last_points = get_lp(self.frame_queues)
            new_queues = enqueue(self.points, last_points, self.frame_queues, self.curr_time, dt)
            for p in self.points:
                self.frame = cv.circle(self.frame, p, radius=10, color=(255, 0, 0), thickness=-1)
        else: # If its empty, we reset the deque
            self.ref_time = time.time()
            new_queues = []

        self.frame_queues = new_queues


# +---------------------------------------+
# | Interpolation
# +---------------------------------------+

class Interpolation_thread:
    def __init__(self, currframe):
        self.frame = currframe
        self.targets = []

        self.frame_queues = []
        self.curr_time = 0
    
    def start(self):
        T = Thread(target = self.interpolation)
        T.start()
        return T
    
    def interpolation(self):
        if not self.frame_queues == []:
            new_targets = get_targets(self.frame, self.frame_queues, self.curr_time)
            self.targets = new_targets
        else:
            self.targets = []



























