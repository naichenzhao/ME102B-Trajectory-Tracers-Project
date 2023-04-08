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

    else:
        last_best, curr_best = get_mapping(last_points, points)
        for i in range(len(last_best)):
            index = last_points.index(last_best[i])
            lasttime = round(frame_queues[index][-1][2], dec)
            # print(lasttime)

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