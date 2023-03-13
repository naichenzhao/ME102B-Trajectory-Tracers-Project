# Import libraries
import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys

# Import other classes
from Get_thread import *
from Show_thread import *






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

    # define a range for the colour red
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    
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
        # | MAIN LOOP
        # +---------------------------------------+

        # get red aspects of the frame
        red_mask_base = cv.inRange(frame_hsv, red_lower, red_upper)
      
        # Create red mask
        kernal = np.ones((5, 5), "uint8")
        red_mask = cv.dilate(red_mask_base, kernal)
        res_red = cv.bitwise_and(frame, frame, mask = red_mask)

        contours, hierarchy = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


        if len(contours) != 0:
            # get largest contour
            cv.drawContours(frame, contours, -1, (0,255,0), 3)
            c_largest = max(contours, key = cv.contourArea)

            # draw rectangle around largest contour
            if cv.contourArea(c_largest) > 300:
                x, y, w, h = cv.boundingRect(c_largest)
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                

        show_fps(get_thread)

    cv.destroyAllWindows()

    print("finished main")


def show_fps(get_thread): 
    print("fps:", get_thread.fps)




if __name__ == '__main__':
    try:
        # Run main program
        main()
    finally:
        print('Program has been interrupted. Terminating all threads')

        # kill OpenCV windows and all threads
        get_thread.stop()
        show_thread.stop()
        cv.destroyAllWindows()

        print('Quitting program')
