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

# Import other classes
from Get_thread import *
from Show_thread import *



MIN_AREA = 1000
DEQUE_SIZE = 50


'''

start: [(607, 346)]

stay: 
    [(575, 388), (424, 311)]
    [(560, 383), (398, 304)]
    [(560, 383), (398, 304)]
    [(531, 379), (368, 298)]
    [(531, 379), (368, 298)]
    [(501, 372), (346, 293)]
    [(501, 372), (346, 293)]
    [(475, 370), (324, 289)]
    [(475, 370), (324, 289)]

leave: [(606, 298), (491, 250)]
       [(625, 279), (540, 267)]
       [(535, 228)]

'''


# +---------------------------------------+
# |  Main Function
# +---------------------------------------+
def main():

    # create queue of most recent points
    frame_queues = []
    
    ref_time = time.time()
    
    ref_points = [(10, 11), (21, 22), (32, 33)]
    ref_point2 = [(11, 11), (22, 22), (33, 33)]

    points =  [(33, 34), (11, 12), (22, 23)]
    
    for i in ref_points:
        frame_queues.append(deque([i], maxlen=DEQUE_SIZE) )
    
    for i in range(len(ref_point2)):
        frame_queues[i].append(ref_point2[i])
    
    
    print(frame_queues)
    last_points = get_lp(frame_queues)

    # If we find a target on the screen
    if points is not []:
        print(points)
        print(last_points)
        print()
        frame_queues = enqueue(points, last_points, frame_queues, ref_time) 

        print(frame_queues)
        

    print("finished main")



def enqueue(points, last_points, frame_queues, ref_time):

    # If the queue is empty
    if frame_queues is []:
        for p in points:
            frame_queues.append(deque([p], maxlen=DEQUE_SIZE))
            return
        
    queue_data = []
    for q in frame_queues:
        queue_data.append(q[0])

    # If a new point appears
    if(len(points) > len(last_points)):
        print(last_points)
        combos = list(combinations(points, len(last_points)))

        curr_best, last_best = get_best_comb(combos, last_points)
        print("last points:", last_best)
        print("start points:", curr_best)

        # Set the previous deques
        for i in range(len(last_best)):
            index = last_points.index(last_best[i]) 
            frame_queues[index].append(curr_best[i])
        
        # add new elements to the array
        diff =  list(set(points) - set(curr_best))
        for i in diff:
            frame_queues.append(deque([i], maxlen=DEQUE_SIZE) )
        return frame_queues

    # If we need to erase an old point 
    elif len(points) < len(last_points) :
        print(points)
        combos = list(combinations(last_points, len(points)))
        print(combos)
        last_best, curr_best = get_best_comb(combos, points)
        print("last points:", last_best)
        print("start points:", curr_best)

        new_queues = []

        # Set the previous deques
        for i in range(len(last_best)):
            index = last_points.index(last_best[i]) 
            frame_queues[index].append(curr_best[i])
            new_queues.append(frame_queues[index])
        
        return new_queues

    else:
        last_best, curr_best = get_mapping(last_points, points)
        for i in range(len(last_best)):
            index = last_points.index(last_best[i]) 
            frame_queues[index].append(curr_best[i])
        
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
     return [a[-1] for a in dequq_arr]

if __name__ == '__main__':
    main()
    