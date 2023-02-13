import cv2 as cv
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from threading import Thread


def main():
    camera = cv.VideoCapture(1)
    while(True):
        camera, frame = camera.read()
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        fps = camera.get(cv.CAP_PROP_FPS)
        print(fps)

        if frame is not None:
            cv.imshow("Frame", frame)
        q = cv.waitKey(1)
        if q == ord("q"):
            break

    cv.destroyAllWindows()

    print("finished main")

if __name__ == '__main__':
    main()
