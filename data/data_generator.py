#%%
import numpy as np
import glob

from freetype import *

from shapely.geometry import LineString, Polygon
from shapely.affinity import rotate, skew
from shapely.ops import transform

from sklearn.model_selection import train_test_split
from IPython.display import display

#%% 
class DatasetGenerator(object):
    def __init__(self, ttfs:list, letters:list):
        self.ttfs = ttfs
        self.letters = letters
        self.geom = []
        self.value = []
                
    def __reflection__(self, x0):
        return lambda x, y: (2*x0 - x, y)
    
    def __interpolate__(self, geom:Polygon):        
        exter_ring = LineString(geom.exterior.coords)       
        exter_len = exter_ring.length   
        exter_coords = []     
        
        step = np.random.randint(10, int(exter_len)//3 + 10)
        # exter ring interpolateion
        for dist in np.arange(1, int(exter_len), step):
            coord = exter_ring.interpolate(dist)
            exter_coords.append((coord.x, coord.y))
            
        inter_rings = list(geom.interiors)
        inter_coords = []
        # inter rings interpolateion
        for inter_ring in inter_rings:
            ring_coords = []
            inter_len = inter_ring.length
            step = np.random.randint(10, int(inter_len)//3 + 10)
            for dist in np.arange(1, int(inter_len), step):
                coord = inter_ring.interpolate(dist)
                ring_coords.append((coord.x, coord.y))
            if len(ring_coords) >= 3:
                inter_coords.append(ring_coords)
        try:
            return Polygon(exter_coords, 
                           inter_coords)
        except ValueError:
            pass
            
    def __augment__(self, geom:Polygon):
    
        return [rotate(geom, np.random.randint(0, 30), 
                       origin='centroid'),
                rotate(geom, np.random.randint(-30, 0), 
                       origin='centroid'),
                skew(geom, xs=np.random.randint(0, 20),
                       origin='centroid'), 
                skew(geom, ys=np.random.randint(0, 20),
                       origin='centroid')]
        
    def __call__(self, 
                 path: str,
                 filename:str, 
                 augment:bool=False):
        for ttf in self.ttfs:
            face = freetype.Face(ttf)
            face.set_char_size(32*64)
            for k, v in self.letters.items():                
                face.load_char(k, freetype.FT_LOAD_DEFAULT 
                       		      | freetype.FT_LOAD_NO_BITMAP)
                slot = face.glyph
                points = slot.outline.points
                contours = slot.outline.contours

                geoms = []
                start_idx = 0
                for idx, end_idx in enumerate(contours):
                    if idx == len(contours):
                        contour = points[start_idx:]
                    else:
                        contour = points[start_idx:end_idx + 1]
                        start_idx = end_idx + 1
                        
                    if len(contour) >= 3:
                        geoms.append(contour)
                            
                geom = Polygon(geoms[0], 
                               [g for g in geoms[1:]])

                eps = 0.5
                if not geom.is_valid:
                    geom = geom.buffer(eps)
                
                if geom.is_empty or not isinstance(geom, Polygon):
                    continue
                
                dense_geoms = [geom]    
                values = [v] * len(dense_geoms)
                    
                # augment dataset
                if augment:
                    aug_geoms = self.__augment__(geom)
                    inter_geom = self.__interpolate__(geom)
                    if inter_geom:
                        aug_geoms.append(inter_geom)
                        
                    dense_geoms += aug_geoms
                    values += [v] * len(aug_geoms)
                    
                self.geom += dense_geoms
                self.value += values
                
        np.savez(path + filename, 
                 geom=self.geom, 
                 value=self.value)
 
path = '../../dataset/'                 
ttfs = glob.glob(path + "ttfs/*")

serif_ttfs = glob.glob(ttfs[0] + '/*') 
serif_ttfs, test_serif_ttfs = train_test_split(serif_ttfs, 
                                               test_size=0.4)

sans_serif_ttfs = glob.glob(ttfs[1] + '/*') 
sans_serif_ttfs, \
test_sans_serif_ttfs = train_test_split(sans_serif_ttfs, 
                                        test_size=0.4)

letters = {'0':0, '1':1, '2':2, '3':3, '4':4, 
           '5':5, '6':6, '7':7, '8':8, '9':9}
# %%
sans_serif_set = DatasetGenerator(sans_serif_ttfs, letters)
sans_serif_set(path + 'LetterVectors/', 
               'sans_serif.npz')
# %%
augmented_sans_serif_set = DatasetGenerator(sans_serif_ttfs, letters)
augmented_sans_serif_set(path + 'LetterVectors/', 
                         'augmented_sans_serif.npz',
                         augment=True)
# %%
test_sans_serif_set = DatasetGenerator(test_sans_serif_ttfs, letters)
test_sans_serif_set(path + 'LetterVectors/', 
                    'test_sans_serif.npz')
#%%  
serif_set = DatasetGenerator(serif_ttfs, letters)
serif_set(path + 'LetterVectors/',
          'serif.npz')
# %%
augmented_serif_set = DatasetGenerator(serif_ttfs, letters)
augmented_serif_set(path + 'LetterVectors/',
                    'augmented_serif.npz', 
                    augment=True)
#%%
test_serif_set = DatasetGenerator(test_serif_ttfs, letters)
test_serif_set(path + 'LetterVectors/',
               'test_serif.npz')
#%%
ttfs = glob.glob(ttfs[0] + '/*') + glob.glob(ttfs[1] + '/*') 
train_ttfs, valid_ttfs = train_test_split(ttfs, test_size=0.4)
valid_ttfs, test_ttfs = train_test_split(valid_ttfs, test_size=0.5)
#%%
train_ttfs_set = DatasetGenerator(train_ttfs, letters)
train_ttfs_set(path + 'LetterVectors/',
               'train_ttfs.npz')
#%%
augmented_train_ttfs_set = DatasetGenerator(train_ttfs, letters)
augmented_train_ttfs_set(path + 'LetterVectors/',
                         'augmented_train_ttfs.npz', 
                         augment=True)
#%%
valid_ttfs_set = DatasetGenerator(valid_ttfs, letters)
valid_ttfs_set(path + 'LetterVectors/',
               'valid_ttfs.npz')
#%%
test_ttfs_set = DatasetGenerator(test_ttfs, letters)
test_ttfs_set(path + 'LetterVectors/',
              'test_ttfs.npz')