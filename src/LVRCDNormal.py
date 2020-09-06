"""
Class for normalizing the sliced images for the LEVIRCD dataset
"""


import numpy as np


# Class to normalize images 
class LVRCDNormal(object):
    """
    class for Normalization of images, per channel, in format CHW 
    """
    def __init__(self):
        
        # Normalization constants for image -- calculated from training images 
        self._mean = np.array([100.90723866,  99.52347812,  84.97354742])
        self._std = np.array ([ 42.8782652 , 40.90759297, 38.31541013 ])


        
    def __call__(self,img):

        temp = img.astype(np.float32)
        temp2 = temp.T            
        temp2 -= self._mean
        temp2 /= self._std
            
        temp = temp2.T

        return temp
        


    def restore(self,normed_img):

        d2 = normed_img.T * self._std
        d2 = d2 + self._mean
        d2 = d2.T
        d2 = np.round(d2)
        d2 = d2.astype('uint8')

        return d2 



