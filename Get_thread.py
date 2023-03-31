from threading import Thread
import cv2 as cv


class Get_thread:
    """ 
    The Get_thread class is used to put data collecting on its own designated thread
        - This should help with processing speed and free up the main thread to 
            focus more on data processing
            
    """
    def __init__(self, stream):
        # connect camera port
        self.stream = stream

        # get first frame
        self.camera, self.frame = self.stream.read()
        self.frame_hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)

        # get fps
        self.fps = self.stream.get(cv.CAP_PROP_FPS)

        # set process to not end
        self.end_task = False

        # counter for testing
        self.counter = 0
    

    def start(self):
        ''' 
        start() initiates the thread to start capturing data
        '''
        Thread(target = self.get_frame).start()
        return self
    

    def get_frame(self):
        ''' 
        get_frame() grabs a new frame from the camera
            - set to run on its own thread to help with processing time
            
        The process will keep running until:
            - Camera is no longer detected
            - Process is manually stopped
        '''
        # Keep going until process is stopped
        while not self.end_task:
            
            if not self.camera:
                # If camera doesnt exist, terminate
                print("Get thread could not detect Camera. Terminating")
                self.stop()
            else:
                # update frames
                self.camera, self.frame = self.stream.read()
                self.frame_hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
                
                # update fps
                self.fps = self.stream.get(cv.CAP_PROP_FPS)
                self.counter = self.counter + 1
                # print("-------- new frame --------")


    def stop(self):
        ''' 
        stop() ends the process and stops camera from grabbing any more frames
        '''
        self.end_task = True