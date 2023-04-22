# Import libraries
import cv2 as cv
import numpy as np
import serial
from collections import *
import time
import sys
import traceback
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt

from threading import Timer

# Import other classes
from utils import *
from Get_thread import *
from Show_thread import *

MIN_AREA = 100
DEQUE_SIZE = 7
MIN_DATA_SIZE = 3

# Get limit of the frame of calculation
X_MAX = 580
X_MIN = 1
Y_MAX = 479
Y_MIN = 1

# Start sending data once ball is past a certain point
SEND_START = 10

# Cup offset
CUP_OFFSET = 42


# Sets limits of stepper
STEPPER_MAX = 8250
STEPPER_MIN = 0


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

arduino = serial.Serial(port='COM8', baudrate=460800, timeout=.1, write_timeout=0)

# +---------------------------------------+
# |  Main Function
# +---------------------------------------+
def main():
    # timer = Timer(20, endTimer)

    # define a range for the colour red
    # red_lower = np.array([60, 110, 170], np.uint8)
    # red_upper = np.array([70, 150, 250], np.uint8)
    green_lower = np.array([50, 120, 120], np.uint8)
    green_upper = np.array([100, 240, 200], np.uint8)

    blue_lower = np.array([100, 240, 130], np.uint8)
    blue_upper = np.array([120, 255, 250], np.uint8)

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
    last_time_sent = time.time()

    
    last_sent = 0

    # initialize deque for  y intercepts
    step_position_deque = deque([0, 0, 0, 0, 0])
    last_position = 0


    # timer.start()
    while (True):

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
        points = find_colour(frame, frame_hsv, green_lower, green_upper)
        cup = find_colour_max(frame, frame_hsv, blue_lower, blue_upper)
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
                inter_t = float(poly_solve(px, X_MAX))
                inter_x = int(px(inter_t))
                inter_y = int(py(inter_t))

                # Save all the viable targets
                if inter_t > curr_time and abs(inter_x) < 1000 and abs(inter_y) < 1000 and x_vals[0] > SEND_START:
                    targets.append((inter_x, inter_y, round(inter_t, 5)))

        # Print out all the x-intercepts
        if len(targets) > 0:
            targets_sorted = sorted(targets, key=lambda x: x[2])

            # Print first to cross X_LIM
            x_first = targets_sorted[0][0]
            y_first = targets_sorted[0][1]

            step_position_desired = int(np.ceil(STEPPER_MAX - STEPPER_MAX * (y_first - Y_MIN) / (Y_MAX - Y_MIN)))

            if step_position_desired > STEPPER_MAX:
                step_position_desired = STEPPER_MAX
            elif step_position_desired < STEPPER_MIN:
                step_position_desired = STEPPER_MIN

            # Removes oldest element and appends new element
            step_position_deque.popleft()
            step_position_deque.append(y_first)
            step_filtered = medfilt(step_position_deque)
            print(y_first)
            last_sent = send_data(cup, min(max(step_filtered[2], 0), 480) , last_sent)


            # First one to cross is highlited in large pink ball
            frame = cv.circle(frame, (x_first, y_first), radius=12, color=(255, 0, 255), thickness=-1)

            # Print out all the targets, sorted in terms of when they pass the intercept
            # print(targets_sorted)

            # print all the other intercepts
            for i in range(1, len(targets_sorted)):
                x_curr = targets_sorted[i][0]
                y_curr = targets_sorted[i][1]
                # All others highlighted in smaller green balls
                frame = cv.circle(frame, (x_curr, y_curr), radius=8, color=(255, 255, 0), thickness=-1)
        else:
            print("centering")
            send_data(cup, 240, 0)

    cv.destroyAllWindows()

    print("finished main")

def send_data(cup, target, last_sent):
    # writes step position_desired
    
    if cup > 0: # Sends data every 500 ms
        # if abs(last_position - step_filtered[3]) > 50: # Sends data only if new posistion is at least 700 steps away from previous
        config = 0
        if cup > target:
            if abs(cup - target) > 50:
                config = 4
            elif abs(cup - target) > 5:
                config = 3
            else:
                config = 0

        elif (cup < target):
            if abs(target - cup) > 50:
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













def find_colour(frame, frame_hsv, red_lower, red_upper):
    # get red aspects of the frame
    c_mask_base = cv.inRange(frame_hsv, red_lower, red_upper)

    # Create red mask
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))

    morph = cv.dilate(c_mask_base, kernel1)

    contours, hierarchy = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame, contours, -1, (0,255,0), 3)

    points = []

    if len(contours) != 0:
        for curr_cnt in contours:
            x, y, w, h = cv.boundingRect(curr_cnt)
            if cv.contourArea(curr_cnt) > MIN_AREA and in_bounds(x, y):
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                points.append((int(x + (w / 2)), int(y + (h / 2))))

    return points

def find_colour(frame, frame_hsv, red_lower, red_upper):
    # get red aspects of the frame
    c_mask_base = cv.inRange(frame_hsv, red_lower, red_upper)

    # Create red mask
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))

    morph = cv.dilate(c_mask_base, kernel1)

    contours, hierarchy = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame, contours, -1, (0,255,0), 3)

    points = []

    if len(contours) != 0:
        for curr_cnt in contours:
            x, y, w, h = cv.boundingRect(curr_cnt)
            if cv.contourArea(curr_cnt) > MIN_AREA and in_bounds(x, y):
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                points.append((int(x + (w / 2)), int(y + (h / 2))))

    return points

def find_colour_max(frame, frame_hsv, lower, upper):
    # get red aspects of the frame
    c_mask_base = cv.inRange(frame_hsv, lower, upper)
      
    # Create red mask
    kernal = np.ones((5, 5), "uint8")
    c_mask = cv.dilate(c_mask_base, kernal)
    res_red = cv.bitwise_and(frame, frame, mask = c_mask)

    contours, hierarchy = cv.findContours(c_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # get largest contour
        c_largest = max(contours, key = cv.contourArea)

        # draw rectangle if it meets the area criteria
        if cv.contourArea(c_largest) > 300:
            x, y, w, h = cv.boundingRect(c_largest)
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            frame = cv.circle(frame, (int(x + (w/2)), int(y+CUP_OFFSET + (h/2)) ), radius=8, color=(255, 0, 0), thickness=-1)
            return int((y+CUP_OFFSET) + (h/2))
    
    return 0



def in_bounds(x, y):
    return (x > X_MIN) and(x < X_MAX) and (y > Y_MIN) and (y < Y_MAX)


def poly_draw(frame, px, py, start, end, res=1000):
    t_points = np.linspace(start, end, 1000)
    x_points = px(t_points)
    y_points = py(t_points)

    for i in range(len(x_points)):
        x = int(x_points[i])
        y = int(y_points[i])
        if x < 0 or y < 0 or x > 1000 or y > 1000:
            continue
        frame = cv.circle(frame, (x, y), radius=1, color=(255, 0, 0), thickness=-1)


def poly_solve(poly, y):
    return (poly - y).roots


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