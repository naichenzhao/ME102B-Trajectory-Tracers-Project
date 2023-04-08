# Import libraries
import cv2 as cv
import numpy as np
from collections import *
import time
import sys
import traceback
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from threading import Timer


# Import other classes
from utils import *
from Get_thread import *
from Show_thread import *



MIN_AREA = 200
DEQUE_SIZE = 8
MIN_DATA_SIZE = 3


# Get limit of the frame of calculation
X_LIM = 600

Y_MAX = 450
Y_MIN = 30





# +---------------------------------------+
# |  General setup
# +---------------------------------------+

# setup video capture stream
cv_stream = cv.VideoCapture(1)

# setup threads for getting and displaying the video stream
get_thread = Get_thread(cv_stream)
get_thread.start()

show_thread = Show_thread(get_thread.frame)
show_thread.start()








# +---------------------------------------+
# |  Main Function
# +---------------------------------------+
def main():

    # timer = Timer(20, endTimer)

    # define a range for the colour red
    red_lower = np.array([60, 170, 150], np.uint8)
    red_upper = np.array([100, 255, 255], np.uint8)


    # Setup other variables
    last_frame = np.zeros((480, 640, 3))

    # any timing variables
    global main_counter
    global newframe_counter

    newframe_counter = 0
    main_counter = 0


    # create queue of most recent points
    frame_queues = []
    ref_time = time.time()
    curr_time = time.time()
    last_time = curr_time


    # timer.start()
    while(True):
         

        # If either of the threads end, stop program
        if get_thread.end_task or show_thread.end_task:
            get_thread.stop()
            show_thread.stop()

            print("[NOTICE]: Terminating all threads")
            break 


        frame = np.copy(get_thread.frame)
        main_counter = main_counter + 1


        # [IMPORTANT] If the frames are the same, we skip this cycle of calculation
        if np.array_equal(get_thread.frame, last_frame):
            continue


        # Show the x_lim cutoff point in orange
        #    Any contous outside of this area will not be counted
        frame = draw_outline(frame)


        # Update time variables
        last_time = curr_time
        curr_time = time.time() - ref_time
        dt = curr_time - last_time

        # Update frame variables
        last_frame = get_thread.frame
        frame_hsv = get_thread.frame_hsv
        show_thread.frame = frame
            
        # Update frame counter (Only for testing)
        newframe_counter = newframe_counter + 1

        # +---------------------------------------+
        # | find colour
        # +---------------------------------------+
        points = find_colour(frame, frame_hsv, red_lower, red_upper)
        # print(curr_time, dt)
        # If we find a target on the screen
        if len(points) != 0:
            last_points = get_lp(frame_queues)
            frame_queues = enqueue(points, last_points, frame_queues, curr_time, dt, DEQUE_SIZE)

            for p in points:
                frame = cv.circle(frame, p, radius=10, color=(255, 0, 0), thickness=-1)

        else:
            # reset everything
            ref_time = time.time()
            frame_queues = []
            
        if len(frame_queues) == 0:
            continue

        targets = []


        for i in frame_queues:
            pos_data = read_queue(i)
            # print(pos_data)
            x_vals = pos_data[:, 0]
            y_vals = pos_data[:, 1]
            t_vals = pos_data[:, 2]

            if len(t_vals) >= MIN_DATA_SIZE:
                zx = np.polyfit(t_vals, x_vals, 1)
                zy = np.polyfit(t_vals, y_vals, 1)
                px = np.poly1d(zx)
                py = np.poly1d(zy)


                # Print out the trajectories of all the balls
                poly_draw(frame, px, py, t_vals[0], t_vals[0] + 10)

                # solve for x-intercepts
                inter_t = float(poly_solve(px, X_LIM))
                inter_x = int(px(inter_t))
                inter_y = int(py(inter_t))

                # Save all the viable targets
                if inter_t > curr_time and abs(inter_x) < 1000 and abs(inter_y) < 1000 :
                    targets.append((inter_x, inter_y, round(inter_t, 5)))
        

        # Print out all the x-intercepts 
        if len(targets) > 0:
            targets_sorted = sorted(targets, key = lambda x:x[2])


            # Print first to cross X_LIM
            x_first = targets_sorted[0][0]
            y_first = targets_sorted[0][1]

            # First one to cross is highlited in large pink ball
            frame = cv.circle(frame, (x_first, y_first), radius=12, color=(255, 0, 255), thickness=-1)

            # Print out all the targets, sorted in terms of when they pass the intercept
            print(targets_sorted)


            # print all the other intercepts
            for i in range(1, len(targets_sorted)):
                x_curr = targets_sorted[i][0]
                y_curr = targets_sorted[i][1]
                # All others highlighted in smaller green balls
                frame = cv.circle(frame, (x_curr, y_curr), radius=8, color=(255, 255, 0), thickness=-1)


            


    cv.destroyAllWindows()

    print("finished main")






def find_colour(frame, frame_hsv, red_lower, red_upper):
    # get red aspects of the frame
    c_mask_base = cv.inRange(frame_hsv, red_lower, red_upper)
      
    # Create red mask
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(12, 12))

    morph = cv.dilate(c_mask_base, kernel1)

    contours, hierarchy = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame, contours, -1, (0,255,0), 3)

    points = []

    if len(contours) != 0:
        for curr_cnt in contours:
            x, y, w, h = cv.boundingRect(curr_cnt)
            if cv.contourArea(curr_cnt) > MIN_AREA and  in_bounds(x, y):
                
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                points.append((int(x + (w/2)), int( y + (h/2))))
    
    return points


def in_bounds(x, y):
    return (x < X_LIM) and (y > Y_MIN) and (y < Y_MAX)

def poly_draw(frame, px, py, start, end, res=1000):
    t_points = np.linspace(start, end, 1000)
    x_points = px(t_points)
    y_points = py(t_points)

    for i in range(len(x_points)):
        x = int(x_points[i])
        y = int(y_points[i])
        if x<0 or y<0 or x>1000 or y>1000:
            continue
        frame = cv.circle(frame, (x, y), radius=1, color=(255, 0, 0), thickness=-1)


def poly_solve(poly, y):

    return (poly - y).roots

def draw_outline(frame):
    
    # Draw X_LIM for where the cup should be
    y_vals = np.linspace(Y_MIN, Y_MAX, 400)
    for i in range(len(y_vals)):
        frame = cv.circle(frame, (X_LIM, int(y_vals[i])), radius=2, color=(0, 150, 255), thickness=-1)

    # Draw Y_MAX and Y_MIN for outlines of the box
    x_vals = np.linspace(0, X_LIM, 800)
    for i in range(len(x_vals)):
        frame = cv.circle(frame, (int(x_vals[i]), Y_MIN), radius=2, color=(0, 150, 255), thickness=-1)
        frame = cv.circle(frame, (int(x_vals[i]), Y_MAX), radius=2, color=(0, 150, 255), thickness=-1)
    return frame


def endTimer():
    print("Timer has ended")
    print("main thread counter:", main_counter)
    print("newframe counter:", newframe_counter)
    print("get thread counter:", get_thread.counter)

    get_thread.stop()
    show_thread.stop()
    cv.destroyAllWindows()
    exit()


if __name__ == '__main__':
    try:
        # Run main program
        main()
    except Exception as e:
        print('Program has been interrupted. Terminating all threads')
        print("Got this error:")
         # Get current system exception
        ex_type, ex_value, ex_traceback = sys.exc_info()

        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        print(e)
        print(trace_back)

        # kill OpenCV windows and all threads
        get_thread.stop()
        show_thread.stop()
        cv.destroyAllWindows()

        print('Quitting program')
