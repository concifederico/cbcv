
import cv2
import numpy as np
import gui
import globvar
"""
An example class of how to implement openCV and how it can communicate with the
wxPython layer.
"""
class WebcamFeed(object):
    
    """ Starts a webcam feed """
    def __init__(self):
        self.webcam = cv2.VideoCapture(globvar.Dispositivo)
        if globvar.RPi == True:
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, int(1280))
            self.webcam.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, int(720))
        else:
            self.webcam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, int(1280))
            self.webcam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, int(720))
		
    """ Determines if the webcam is available """
    def has_webcam(self):
        _, frame = self.webcam.read()
        if(isinstance(frame, np.ndarray)):
            return True
        return False
    
    """ Retrieves a frame from the webcam and converts it to an RGB - Image """
    def get_image(self, w=None, h=None):
        _, frame = self.webcam.read()
        if w != None and h != None:
            frame = cv2.resize(frame, (w, h))
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    """ Retrieves a frame to get the size """
    def size(self):
        _, frame = self.webcam.read()
        return frame.shape[:2]
