# Import libraries
import cv2 as cv
import numpy as np
from collections import *
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# +---------------------------------------+
# |  Declare any constants needed
# +---------------------------------------+

# Interpolation function constants
DEQUE_SIZE = 7
MIN_DATA_SIZE = 3

# Minimum area for recognizing a blob
MIN_AREA = 100

# Get limit of the frame of calculation
X_MAX = 580
X_MIN = 1
Y_MAX = 479
Y_MIN = 1

# Start sending data once ball is past a certain point
SEND_START = 10

# Cup offset
CUP_OFFSET = 42


# +----------------------------------------------------------------------------+
# |
# |                             [Colour Recognition]
# |
# +----------------------------------------------------------------------------+

def find_points_all(frame, frame_hsv, lower, upper):
    # get red aspects of the frame
    c_mask_base = cv.inRange(frame_hsv, lower, upper)

    # Dialate values to get rid of small defects
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
    morph = cv.dilate(c_mask_base, kernel1)
    contours, hierarchy = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    points = []
    if len(contours) != 0:
        for curr_cnt in contours:
            x, y, w, h = cv.boundingRect(curr_cnt)
            if cv.contourArea(curr_cnt) > MIN_AREA and in_bounds(x, y):
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                points.append((int(x + (w / 2)), int(y + (h / 2))))

    return points

def find_points_max(frame, frame_hsv, lower, upper):
    # get red aspects of the frame
    c_mask_base = cv.inRange(frame_hsv, lower, upper)

    # Dialate values to get rid of small defects
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (12, 12))
    morph = cv.dilate(c_mask_base, kernel1)
    contours, hierarchy = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # get largest contour
        c_largest = max(contours, key = cv.contourArea)

        # draw rectangle if it meets the area criteria
        if cv.contourArea(c_largest) > 300:
            x, y, w, h = cv.boundingRect(c_largest)

            x_center = int(x + (w/2))
            y_center = int(y + CUP_OFFSET + (h/2))

            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            frame = cv.circle(frame, (x_center, y_center), radius=8, color=(255, 0, 0), thickness=-1)
            return y_center
    
    # If a cup has not been found
    return 0

def in_bounds(x, y):
    return (x > X_MIN) and(x < X_MAX) and (y > Y_MIN) and (y < Y_MAX)






# +----------------------------------------------------------------------------+
# |
# |                       [Point Continuity and Enqueue]
# |
# +----------------------------------------------------------------------------+

def enqueue(points, last_points, frame_queues, currtime, dt, DEQUE_SIZE):
    dec = 5
    # if no new points
    if len(points) == 0:
        return []

    # If the queue is empty
    if len(frame_queues) == 0:
        for p in points:
            frame_queues.append(deque([(p[0], p[1], round(currtime, dec))], maxlen=DEQUE_SIZE))
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
            lasttime = round(frame_queues[index][-1][2], dec)

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
            lasttime = round(frame_queues[index][-1][2], dec)

            newpoint = (curr_best[i][0], curr_best[i][1], lasttime+dt)
            frame_queues[index].append(newpoint)
            new_queues.append(frame_queues[index])
        
        return new_queues

    # If the number of points match up
    else:
        last_best, curr_best = get_mapping(last_points, points)
        for i in range(len(last_best)):
            index = last_points.index(last_best[i])
            lasttime = round(frame_queues[index][-1][2], dec)
            # print(lasttime)

            newpoint = (curr_best[i][0], curr_best[i][1], lasttime+dt)
            frame_queues[index].append(newpoint)
        
        return frame_queues
        

# Gets the bext combination of points (minimum total length)
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

# Gets the cost of a configuration (Sum of legths)
def get_cost(points1, points2):
    return np.sum(cdist(points1, points2) * np.eye(len(points1)))

# Get the optimal mapping between two lists of points
def get_mapping(points1, points2):
    C = cdist(points1, points2)
    _, b = linear_sum_assignment(C)
    new_points = []
    for i in range(len(points2)):
        new_points.append(points2[b[i]])

    return points1, new_points

# Get the last points from a queue
def get_lp(dequq_arr):
     return [(a[-1][0], a[-1][1]) for a in dequq_arr]

# Reads a dequeue and returns an array of values
def read_queue(d):
    deque_length = len(list(d))
    if deque_length == 0:
       return None

    framedata = np.zeros((deque_length, 3))

    for i in range(deque_length):
        framedata[i] = d[i]
    
    return framedata






# +----------------------------------------------------------------------------+
# |
# |                             [Interpolation]
# |
# +----------------------------------------------------------------------------+
def get_targets(frame, frame_queues, curr_time):
    targets = []
    for i in frame_queues:
        pos_data = read_queue(i)
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
    
    if len(targets) > 0:
        return sorted(targets, key=lambda x: x[2])
    else:
        return targets

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


