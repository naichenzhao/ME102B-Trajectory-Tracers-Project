# Import libraries
import cv2 as cv
import numpy as np
import serial
import time
import sys
import traceback


from scipy.signal import medfilt
from collections import *
from threading import Timer
from threading import Thread
import threading

# Import other python files
from utils import *
from Get_thread import *
from Show_thread import *
from Process_thread import *


'''
    +---------------------------------------+
    |  PROJECT ALGORITHM OUTLINE
    +---------------------------------------+

    The project algorith is broken up into a number of main chunks

    [Image Fetch] --> [Colour Recognition] --> [Point Continuity and Enqueue] --> [Interpolation] --> [Data filtering] --> [Data Send]

        [Image Fetch]
            - Fetches the image through OpenCV (This is done on the get_thread)
        
        [Colour Recognition]
            - Locates the colour blobs from the frame
                Green - Balls       Blue - cup
        
        [Point Continuity and Enqueue]
            - Compares current point locations to previous point locations to determing continuity betreen frames
            - Then, adds point data to the deque array

        [Interpolation]
            - Runs least-squares on the location history of the balls to create functions for x and y as a function of time
            - Determines both the time and the y-value for when the ball will cross the X_MAX line
        
        [Data Filtering]
            - Sorts the points in terms of which will cross the X_MAX line first
            - Applies a median filter on the expected y-values to get rid of any outliers
        
        [Data Send]
            - Sends movement data to the ESP32
'''


# +---------------------------------------+
# |  General setup
# +---------------------------------------+

# define a range for the vall colour
green_lower = np.array([50, 120, 70], np.uint8)
green_upper = np.array([100, 255, 200], np.uint8)
green_bounds = [green_lower, green_upper]

# Define the range for the cup colour
# blue_lower = np.array([100, 240, 130], np.uint8)
# blue_upper = np.array([120, 255, 250], np.uint8)
blue_lower = np.array([90, 70, 20], np.uint8)
blue_upper = np.array([120, 160, 50], np.uint8)
blue_bounds = [blue_lower, blue_upper]



# Setup video capture stream
cv_stream = cv.VideoCapture(1)

# Setup threads for getting and displaying the video stream
get_thread = Get_thread(cv_stream)
get_thread.start()

show_thread = Show_thread(get_thread.frame)
show_thread.start()

# Setup threads for main function
CR_thread = Colour_Rec_thread(get_thread.frame, get_thread.frame_hsv, green_bounds, blue_bounds)
CE_thread = Enqueue_thread(get_thread.frame)
I_thread = Interpolation_thread(get_thread.frame)

# Add communication port with arduino. If none exists, we ignore it
try:
    arduino = serial.Serial(port='COM8', baudrate=460800, timeout=.1, write_timeout=0)
except Exception as e:
    print("Failed to attach Arduino port... Continuing nonetheless")
    arduino = None


# +---------------------------------------+
# |  Main Function
# +---------------------------------------+
def main():

    # Setup other variables
    last_frame = np.zeros((480, 640, 3))
    frame = np.copy(get_thread.frame)
    frame_CE = frame.copy()
    frame_I = frame_CE.copy()

    # initialize deque for  y intercepts
    step_position_deque = deque([0, 0, 0, 0, 0], maxlen=5)

    # Initialize variables
    points = []
    frame_queues = []
    curr_time = []

    frame_fire = True
    frame_fire_CE = True
    frame_fire_I = True

    while (True):
        # If either of the threads end, stop program
        if get_thread.end_task or show_thread.end_task:
            get_thread.stop()
            show_thread.stop()

            print("[NOTICE]: Terminating all threads")
            break

        # +---------------------------------------+
        # | Frame Fetch
        # +---------------------------------------+
        
        # [IMPORTANT] If the frames are the same, we skip this cycle of calculation
        frame_fire = not np.array_equal(get_thread.frame, last_frame)

        if not (frame_fire or frame_fire_CE or frame_fire_I):
            continue

        # Draw out the outline of what we can see
        #    Any contous outside of this area will not be counted
        if frame_fire:
            frame = np.copy(get_thread.frame)
            frame = draw_outline(frame)
            
            # Update frame variables
            last_frame = get_thread.frame
            frame_hsv = get_thread.frame_hsv
        
        

        # +---------------------------------------+
        # | Main Loop
        # +---------------------------------------+

        
        CR_thread.frame = frame
        CR_thread.frame_hsv = frame_hsv

        CE_thread.frame = frame_CE
        CE_thread.points = points

        I_thread.frame = frame_I
        I_thread.frame_queues = frame_queues
        I_thread.curr_time = curr_time

        # Run functions
        if frame_fire:
            start_CR = CR_thread.start()

        if frame_fire_CE:
            start_CE = CE_thread.start()

        if frame_fire_I:
            start_I = I_thread.start()

        # Wait for loops to finish
        start_CR.join()
        start_CE.join()
        start_I.join()

        # Read Values
        if frame_fire:
            frame_CE = frame.copy()
            points = CR_thread.points
            cup = CR_thread.cup

        if frame_fire_CE:
            frame_I = frame_CE.copy()
            frame_queues = CE_thread.frame_queues
            curr_time = CE_thread.curr_time

        if frame_fire_I:
            targets = I_thread.targets
            if len(targets) > 0:
                draw_intercepts(targets, frame_I)      
        
        # Set the show frame
        show_thread.frame = frame_I

        # Update pipeline frame values
        frame_fire_I = frame_fire_CE
        frame_fire_CE = frame_fire

        
        
        # +---------------------------------------+
        # | Data Filtering
        # +---------------------------------------+
        if len(targets) > 0: # If there are targets, we move the cup
            # Print first to cross X_LIM
            y_target = targets[0][1]

            # Removes oldest element and appends new element
            step_position_deque.append(y_target)
            step_filtered = medfilt(step_position_deque)
            send_data(cup, min(max(step_filtered[2], 0), 480))

        else: # If there are no targets, we center ourselves
            # print("centering")
            send_data(cup, 240)


    cv.destroyAllWindows()
    print("finished main")


def send_data(cup, target):
    # writes step position_desired
    if cup > 0:  # Sends data every 500 ms
        # if abs(last_position - step_filtered[3]) > 50: # Sends data only if new posistion is at least 700 steps away from previous
        config = 0
        if cup < CUP_MIN:
            config = 1
        elif cup > CUP_MAX:
            config = 4
        elif cup > target:
            if abs(cup - target) > 100:
                config = 5
            elif abs(cup - target) > 50:
                config = 4
            elif abs(cup - target) > 5:
                config = 3
            else:
                config = 0
        elif cup < target:
            if abs(target - cup) > 100:
                config = 6
            elif abs(target - cup) > 50:
                config = 1
            elif abs(target - cup) > 5:
                config = 2
            else:
                config = 0

        # if last_sent != config:

        write(config)
        print("c:", cup, "t:", target, "config:", config)
        return config

        # last_position =  target
        # last_time_sent = time.time()


# Writes to serial line
def write(x):
    y = str(x)
    if arduino is not None:
        arduino.write(bytes(y, 'utf-8'))




# +---------------------------------------+
# |  Printing data helper functions
# +---------------------------------------+

def draw_intercepts(targets, frame):
    # First one to cross is highlited in large pink ball
    x_first = targets[0][0]
    y_first = targets[0][1]
    frame = cv.circle(frame, (x_first, y_first), radius=12, color=(255, 0, 255), thickness=-1)

    # print all the other intercepts
    for i in range(1, len(targets)):
        x_curr = targets[i][0]
        y_curr = targets[i][1]

        # All others highlighted in smaller green balls
        frame = cv.circle(frame, (x_curr, y_curr), radius=8, color=(255, 255, 0), thickness=-1)


def draw_outline(frame):
    # Draw X_LIM for where the cup should be
    y_vals = np.linspace(Y_MIN, Y_MAX, 400)
    for i in range(len(y_vals)):
        frame = cv.circle(frame, (X_MAX, int(y_vals[i])), radius=2, color=(0, 150, 255), thickness=-1)
        frame = cv.circle(frame, (X_MIN, int(y_vals[i])), radius=2, color=(0, 150, 255), thickness=-1)
        frame = cv.circle(frame, (SEND_START, int(y_vals[i])), radius=2, color=(150, 150, 255), thickness=-1)
    
    # Draw Y_MAX and Y_MIN for outlines of the box
    x_vals = np.linspace(0, X_MAX, 800)
    for i in range(len(x_vals)):
        frame = cv.circle(frame, (int(x_vals[i]), Y_MIN), radius=2, color=(0, 150, 255), thickness=-1)
        frame = cv.circle(frame, (int(x_vals[i]), Y_MAX), radius=2, color=(0, 150, 255), thickness=-1)
        frame = cv.circle(frame, (int(x_vals[i]), int((Y_MAX-Y_MIN)/2 + Y_MIN)), radius=2, color=(0, 150, 255), thickness=-1)
        
    return frame




# +---------------------------------------+
# |  Main Call
# +---------------------------------------+
if __name__ == '__main__':

    main()
    print('Quitting program')
