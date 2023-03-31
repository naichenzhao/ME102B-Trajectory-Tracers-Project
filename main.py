# Import libraries
import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import *
import time
import sys
import traceback
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from threading import Timer

# Import other classes
from Get_thread import *
from Show_thread import *



MIN_AREA = 1000
DEQUE_SIZE = 10
MIN_DATA_SIZE = 2



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
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)


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

        last_time = curr_time
        curr_time = time.time() - ref_time
        dt = curr_time - last_time


        frame = np.copy(get_thread.frame)
        main_counter = main_counter + 1


        # [IMPORTANT] If the frames are the same, we skip this cycle of calculation
        if np.array_equal(get_thread.frame, last_frame):
            continue
        last_frame = get_thread.frame
        frame_hsv = get_thread.frame_hsv
        show_thread.frame = frame
            

        newframe_counter = newframe_counter + 1

        # +---------------------------------------+
        # | find colour
        # +---------------------------------------+
        points = find_colour(frame, frame_hsv, red_lower, red_upper)

        # If we find a target on the screen
        if points is not []:
            last_points = get_lp(frame_queues)
            frame_queues = enqueue(points, last_points, frame_queues, curr_time, dt)

            for p in points:
                frame = cv.circle(frame, p, radius=10, color=(255, 0, 0), thickness=-1)

        else:
            # reset everything
            ref_time = time.time()
            frame_queue = []
            
        if len(frame_queues) == 0:
            continue


        for i in frame_queues:
            pos_data = read_queue(i)
            x_vals = pos_data[:, 0]
            y_vals = pos_data[:, 1]
            t_vals = pos_data[:, 2]

            if len(x_vals) >= MIN_DATA_SIZE:
                zx = np.polyfit(t_vals, x_vals, 1)
                zy = np.polyfit(t_vals, y_vals, 1)
                px = np.poly1d(zx)
                py = np.poly1d(zy)

                t_points = np.linspace(t_vals[0], t_vals[0] + 0.1, 1000)
                x_points = px(t_points)
                y_points = py(t_points)

                for i in range(len(x_points)):
                    x = int(x_points[i])
                    y = int(y_points[i])
                    if x<0 or y<0 or x>1000 or y>1000:
                        continue
                    try:
                        frame = cv.circle(frame, (x, y), radius=2, color=(255, 0, 0), thickness=-1)
                    except Exception as e:
                        print("[Error with set frame]")
                        print(x, y)


    cv.destroyAllWindows()

    print("finished main")




def enqueue(points, last_points, frame_queues, currtime, dt):
    # if no new points
    if len(points) == 0:
        return []

    # If the queue is empty
    if len(frame_queues) == 0:
        for p in points:
            frame_queues.append(deque([(p[0], p[1], round(currtime, 2))], maxlen=DEQUE_SIZE))
        return frame_queues
        
    queue_data = []
    for q in frame_queues:
        queue_data.append(q[0])

    # If a new point appears
    if(len(points) > len(last_points)):
        combos = list(combinations(points, len(last_points)))

        curr_best, last_best = get_best_comb(combos, last_points)

        # Set the previous deques
        for i in range(len(last_best)):
            index = last_points.index(last_best[i])
            lasttime = round(frame_queues[index][-1][2], 2)

            newpoint = (curr_best[i][0], curr_best[i][1], lasttime+dt)
            frame_queues[index].append(newpoint)
        
        # add new elements to the array
        diff =  list(set(points) - set(curr_best))
        for i in diff:
            frame_queues.append(deque([(i[0], i[1], round(currtime, 2))], maxlen=DEQUE_SIZE))
        return frame_queues

    # If we need to erase an old point 
    elif len(points) < len(last_points) :
        combos = list(combinations(last_points, len(points)))
        last_best, curr_best = get_best_comb(combos, points)

        new_queues = []

        # Set dequeue items (without non-updated ones)
        for i in range(len(last_best)):
            index = last_points.index(last_best[i])
            lasttime = round(frame_queues[index][-1][2], 2)

            newpoint = (curr_best[i][0], curr_best[i][1], lasttime+dt)
            frame_queues[index].append(newpoint)
            new_queues.append(frame_queues[index])
        
        return new_queues

    else:
        last_best, curr_best = get_mapping(last_points, points)
        for i in range(len(last_best)):
            index = last_points.index(last_best[i])
            lasttime = round(frame_queues[index][-1][2], 2)

            newpoint = (curr_best[i][0], curr_best[i][1], lasttime+dt)
            frame_queues[index].append(newpoint)
        
        return frame_queues
        


def get_best_comb(combos, points):
    sums = []
    points1 = []
    points2 = []
    for comb in combos:
        point1, point2 = get_mapping(comb, points)
        sums.append(get_cost(point1, point2))
        points1.append(point1)
        points2.append(point2)

    index = sums.index(min(sums)) 
    return list(points1[index]), list(points2[index])


def get_cost(points1, points2):
    return np.sum(cdist(points1, points2) * np.eye(len(points1)))


def get_mapping(points1, points2):
    C = cdist(points1, points2)
    _, b = linear_sum_assignment(C)
    new_points = []

    for i in range(len(points2)):
        new_points.append(points2[b[i]])

    return points1, new_points

def get_lp(dequq_arr):
     return [(a[-1][0], a[-1][1]) for a in dequq_arr]



def read_queue(d):
    deque_length = len(list(d))
    if deque_length == 0:
       return None

    framedata = np.zeros((deque_length, 3))

    for i in range(deque_length):
        framedata[i] = d[i]
    
    return framedata


def find_colour(frame, frame_hsv, red_lower, red_upper):
    # get red aspects of the frame
    c_mask_base = cv.inRange(frame_hsv, red_lower, red_upper)
      
    # Create red mask
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT,(12, 12))

    morph = cv.dilate(c_mask_base, kernel1)

    contours, hierarchy = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame, contours, -1, (0,255,0), 3)

    points = []

    if len(contours) != 0:
        for curr_cnt in contours:
            if cv.contourArea(curr_cnt) > MIN_AREA:
                x, y, w, h = cv.boundingRect(curr_cnt)
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                points.append((int(x + (w/2)), int( y + (h/2))))
    
    return points


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
