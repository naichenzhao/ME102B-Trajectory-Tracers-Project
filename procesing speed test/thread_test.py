import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys
from threading import Timer

# Import other classes
from Get_thread import *
from Show_thread import *


# +---------------------------------------+
# |  General setup
# +---------------------------------------+

# setup video capture stream
cv_stream = cv.VideoCapture(1)

# define a range for the colour red
red_lower = np.array([136, 87, 111], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)

get_thread = Get_thread(cv_stream)
get_thread.start()

show_thread = Show_thread(get_thread.frame)
show_thread.start()

    
# +---------------------------------------+
# |  Main Function
# +---------------------------------------+
def main_allthread():
    # setup threads for getting and displaying the video stream
    timer = Timer(60, endTimer)
    global main_counter
    main_counter = 0

    timer.start()
    while(True):    
        # If either of the threads end, stop program
        if get_thread.end_task or show_thread.end_task:
            get_thread.stop()
            show_thread.stop()

            print("[NOTICE]: Terminating all threads")
            break
        
        # Update frames
        frame = get_thread.frame
        frame_hsv = get_thread.frame_hsv
        show_thread.frame = frame

        # +---------------------------------------+
        # | find colour
        # +---------------------------------------+
        find_colour(frame, frame_hsv, red_lower, red_upper)
        main_counter = main_counter + 1
        # show_fps(get_thread)

    cv.destroyAllWindows()
    print("finished main")



def main_getthread():
    # setup threads for getting and displaying the video stream
    timer = Timer(60, endTimer)
    global main_counter
    main_counter = 0

    timer.start()
    while(True):
        # If either of the threads end, stop program
        if get_thread.end_task:
            get_thread.stop()
            print("[NOTICE]: Terminating all threads")
            break

        # Update frames
        frame = get_thread.frame
        frame_hsv = get_thread.frame_hsv 
        
        # +---------------------------------------+
        # | find colour
        # +---------------------------------------+
        find_colour(frame, frame_hsv, red_lower, red_upper)
        main_counter = main_counter + 1

        cv.imshow("Frame", frame)
        if cv.waitKey(1) == ord("q"):
            break

    cv.destroyAllWindows()
    print("finished main")



def main_st():
    # setup threads for getting and displaying the video stream
    timer = Timer(60, endTimer)
    global main_counter
    main_counter = 0

    timer.start()
    while(True):
        camera, frame = cv_stream.read()
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)    
        
        # +---------------------------------------+
        # | find colour
        # +---------------------------------------+
        find_colour(frame, frame_hsv, red_lower, red_upper)
        main_counter = main_counter + 1

        cv.imshow("Frame", frame)
        if cv.waitKey(1) == ord("q"):
            break

    cv.destroyAllWindows()

    print("finished main")



def find_colour(frame, frame_hsv, red_lower, red_upper):

    # get red aspects of the frame
    c_mask_base = cv.inRange(frame_hsv, red_lower, red_upper)
      
    # Create red mask
    kernal = np.ones((5, 5), "uint8")
    c_mask = cv.dilate(c_mask_base, kernal)
    res_red = cv.bitwise_and(frame, frame, mask = c_mask)

    contours, hierarchy = cv.findContours(c_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    if len(contours) != 0:
        # get largest contour
        cv.drawContours(frame, contours, -1, (0,255,0), 3)
        c_largest = max(contours, key = cv.contourArea)

        # draw rectangle around largest contour
        if cv.contourArea(c_largest) > 300:
            x, y, w, h = cv.boundingRect(c_largest)
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)



def show_fps(get_thread): 
    print("fps:", get_thread.fps)



def endTimer():
    print("Timer has ended")
    print("main thread counter:", main_counter)
    print("get thread counter:", get_thread.counter)
    print("show thread counter:", show_thread.counter)

    get_thread.stop()
    show_thread.stop()
    cv.destroyAllWindows()
    exit()





if __name__ == '__main__':
    main_allthread()

