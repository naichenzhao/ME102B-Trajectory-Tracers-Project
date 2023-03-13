from threading import Thread
import cv2 as cv


class Show_thread:
    """ 
    The Show_thread class is used to display the data capturing stream
        - This should help with processing speed and free up the main thread to 
            focus more on data processing  

    """

    def __init__(self, curr_frame = None):
        # grabs the current frame and saves it
        self.frame = curr_frame

        # Sets task not to end
        self.end_task = False

        # counter for testing
        self.counter = 0
    

    def start(self):
        ''' 
        start() initiates the thread to start displaying frames

        '''
        Thread(target=self.show_frame).start()
        return self
    

    def show_frame(self):
        ''' 
        show_frame() displays the most recent frame
            - set to run on its own thread to help with processing time
            
        Note: This seems to run into issues while run on MacOS

        The process will keep running until:
            - the user presses "q" to end task
            - Process is manually stopped

        '''
        while not self.end_task:
            # Show current frame
            cv.imshow("Frame", self.frame)
            self.counter = self.counter + 1

            # End stream if necessary
            if cv.waitKey(1) == ord("q"):
                self.end_task = True


    def stop(self):
        '''
        stop() ends the process and stops displaying the frames

        '''
        self.end_task= True