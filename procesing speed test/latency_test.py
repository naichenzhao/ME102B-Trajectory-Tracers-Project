# Import libraries
import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import sys
from threading import Timer
import time

import pygame


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

# setup threads for getting and displaying the video stream
get_thread = Get_thread(cv_stream)
get_thread.start()

show_thread = Show_thread(get_thread.frame)
show_thread.start()

surface = pygame.display.set_mode((400, 300))
 
# Initializing RGB Color
color_red = (255, 0, 0)
color_green = (0, 255, 0)
 
# Changing surface color
surface.fill(color_green)
pygame.display.flip()




    
# +---------------------------------------+
# |  Main Function
# +---------------------------------------+
def main_allthread():
    # setup threads for getting and displaying the video stream
    # timer = Timer(60, endTimer)
    global main_counter
    global start_time
    global endtest


    main_counter = 0
    start_time = 0
    endtest = 0
    result = False

    # timer.start()
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

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    surface.fill(color_red)
                    pygame.display.flip()

                    # start counters
                    start_time = time.time_ns()
                    main_counter = 0

            


        # +---------------------------------------+
        # | find colour
        # +---------------------------------------+
        lastresult = result
        result = find_colour(frame, frame_hsv, red_lower, red_upper)
        main_counter = main_counter + 1

        if result and not lastresult:
            t_diff = time.time_ns() - start_time
            print("time difference in ns:", t_diff)
            print("cycle count is:", main_counter)
            print()

            surface.fill(color_green)
            pygame.display.flip()

            

        # show_fps(get_thread)

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
            return True
        return False



def show_fps(get_thread): 
    print("fps:", get_thread.fps)



def endProgram():
    get_thread.stop()
    show_thread.stop()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main_allthread()

