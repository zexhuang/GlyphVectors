from cv2 import rotate
from numpy import mat
import torch
import math

class ToCoordinate(object):    
    def __call__(self, sample):
        coordinate = sample[0:2, :-1]
        return coordinate
    
# Roate around the center of polygon
class Rotation(object):
    def __init__(self, degrees):
        self.degress = degrees
    
    def __call__(self, sample):
        degree = math.pi * self.degress / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        
        rotate_matrix = [[cos, sin], [-sin, cos]]
        rotated_sample = sample @ rotate_matrix
        return rotated_sample