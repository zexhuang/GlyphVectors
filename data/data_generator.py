
#%% 
import numpy as np
import glob
import matplotlib.pyplot as plt

from freetype import *
from shapely.geometry import LineString
from shapely.affinity import rotate
from shapely.ops import transform
from sklearn.model_selection import train_test_split

class DatasetGenerator(object):
    def __init__(self, ttfs:list, letters:list):
        self.ttfs = ttfs
        self.letters = letters
        self.dataset = []
                
    def __reflection__(self, x0):
        return lambda x, y: (2*x0 - x, y)
    
    def __augment__(self, sample:dict):
        sample_dict = {}
        # for sample in self.dataset:
        geom = LineString(sample['geom'])
        sample_dict['char'] = sample['char']
        # rotation
        r_geom = rotate(geom, 120, origin='centroid')
        sample_dict['type'] = 'r'
        sample_dict['geom'] = np.array(r_geom.xy).T 
        self.dataset.append(sample_dict)
        
        rr_geom = rotate(r_geom, 120, origin='centroid')
        sample_dict['type'] = 'rr'
        sample_dict['geom'] = np.array(rr_geom.xy).T 
        self.dataset.append(sample_dict)
        # reflection
        rf_geom = transform(self.__reflection__(1), geom)
        sample_dict['type'] = 'rf'
        sample_dict['geom'] = np.array(rf_geom.xy).T 
        self.dataset.append(sample_dict)
        # reflection & rotation
        rfr_geom = rotate(rf_geom, 120, origin='centroid')
        sample_dict['type'] = 'rfr'
        sample_dict['geom'] = np.array(rfr_geom.xy).T 
        self.dataset.append(sample_dict)
        
        rfrr_geom = rotate(rfr_geom, 120, origin='centroid')
        sample_dict['type'] = 'rfrr'
        sample_dict['geom'] = np.array(rfrr_geom.xy).T 
        self.dataset.append(sample_dict)
    
    def __interpolate__(self, line:LineString, 
                              char:str,
                              augment:bool):
        length = line.length
        for step in list(range(50, 101, 1)):
            sample_dict = {}
            tem_coords = []
            for dist in np.arange(0, int(length), step):
                coord = line.interpolate(dist)
                tem_coords.append((coord.x, coord.y))
                
            sample_dict['type'] = 'i'
            sample_dict['geom'] = np.array(tem_coords)
            sample_dict['char'] = char
            self.dataset.append(sample_dict)
            
            if augment:
                self.__augment__(sample_dict)
                
    def __call__(self, filename:str, 
                       interpolate:bool=False,
                       augment:bool=False):
        for ttf in self.ttfs:
            face = freetype.Face(ttf)
            face.set_char_size(24*24)
            for k, v in self.letters.items():                
                face.load_char(k, freetype.FT_LOAD_DEFAULT 
                       		       | freetype.FT_LOAD_NO_BITMAP)
                slot = face.glyph
                points = slot.outline.points
                # append new sample to dataset
                sample_dict = {}
                sample_dict['type'] = 'o'
                sample_dict['geom'] = np.array(points)
                sample_dict['char'] = v
                
                # augment dataset
                if augment:
                    self.__augment__(sample_dict)
                    
                # interpolate new points 
                line = LineString(points)
                if interpolate:
                    self.__interpolate__(line, v, augment)
                    
        np.savez(filename, data=self.dataset)
                                   
ttfs = glob.glob("../../dataset/ttfs/*.ttf")
train_ttfs, val_ttfs = train_test_split(ttfs, test_size=0.33, random_state=50) 
letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E':4, 'F':5, 
           'G':6, 'H': 7, 'I':8, 'J':9, 'K': 10, 'L':11, 
           'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17,
           'S':18, 'T':19, 'U': 20, 'V': 21, 'W':22, 'X': 23, 
           'Y':24, 'Z': 25}

train_set = DatasetGenerator(train_ttfs, letters)
train_set('../../dataset/LetterVectors/train_augmented.npz', 
          interpolate=True, 
          augment=True)
val_set = DatasetGenerator(val_ttfs, letters)
val_set('../../dataset/LetterVectors/val_augmented.npz', 
          interpolate=True, 
          augment=True)
