# Import libraries
import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import *
import time
import sys
import traceback

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


    # Setup other variables
    last_frame = np.zeros((480, 640, 3))


    # create queue of most recent points
    DEQUE_SIZE = 50
    frame_queue = deque(DEQUE_SIZE*[[0, 0, 0]], maxlen=DEQUE_SIZE) 
    ref_time = time.time()
    
    while(True):
        # If either of the threads end, stop program
        if get_thread.end_task or show_thread.end_task:
            get_thread.stop()
            show_thread.stop()

            print("[NOTICE]: Terminating all threads")
            break

        
        frame = get_thread.frame
        if  not np.array_equal(frame, last_frame):
            last_frame = frame
            frame_hsv = get_thread.frame_hsv
            show_thread.frame = frame

            # +---------------------------------------+
            # | find colour
            # +---------------------------------------+
            x, y = find_colour(frame, frame_hsv, red_lower, red_upper)

            # If we find a target on the screen
            if x is not None:
                frame = cv.circle(frame, (x,y), radius=10, color=(255, 0, 0), thickness=-1)

                newpoint = [x, y, round(time.time() - ref_time, 2)]
                frame_queue.append(newpoint)

            else:
                
                # reset everything
                ref_time = time.time()
                frame_queue.clear()
            

            pos_data = read_quque(frame_queue)

            if pos_data is  None:
                continue
            x_vals = pos_data[:, 0]
            y_vals = pos_data[:, 1]
            t_vals = pos_data[:, 2]
            print(t_vals)

            if len(x_vals) > 2:
                zx = np.polyfit(t_vals, x_vals, 1)
                zy = np.polyfit(t_vals, y_vals, 2)
                px = np.poly1d(zx)
                py = np.poly1d(zy)

                t_points = np.linspace(t_vals[0]-0.02, t_vals[0] + 2, 3000)
                x_points = px(t_points)
                y_points = py(t_points)

                for i in range(len(x_points)):
                    x = int(x_points[i])
                    y = int(y_points[i])
                    frame = cv.circle(frame, (x, y), radius=1, color=(255, 0, 0), thickness=-1)







            



    cv.destroyAllWindows()

    print("finished main")


def show_fps(get_thread): 
    print("fps:", get_thread.fps)

def read_quque(d):
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
    kernal = np.ones((5, 5), "uint8")
    c_mask = cv.dilate(c_mask_base, kernal)
    res_red = cv.bitwise_and(frame, frame, mask = c_mask)

    contours, hierarchy = cv.findContours(c_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # get largest contour
        cv.drawContours(frame, contours, -1, (0,255,0), 3)
        c_largest = max(contours, key = cv.contourArea)

        # draw rectangle if it meets the area criteria
        if cv.contourArea(c_largest) > 300:
            x, y, w, h = cv.boundingRect(c_largest)
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            return int(x + (w/2)), int( y + (h/2))
    
    return None, None




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
