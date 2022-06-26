from typing import Iterable
import numpy as np
import pandas as pd
import glob

import freetype
from shapely.geometry import Polygon
from shapely.affinity import rotate, skew, scale
from sklearn.model_selection import train_test_split
from IPython.display import display


class Rotate(object):
    def __init__(self):
        pass

    def __call__(self, geom):
        deg = np.random.randint(-60, 60)
        return rotate(geom, deg, origin='centroid')


class Deform(object):
    def __init__(self):
        pass

    def __call__(self, geom):
        xs = np.random.randint(-45, 45)
        ys = np.random.randint(-45, 45)
        return skew(geom, xs=xs, ys=ys, origin='centroid')


class Scale(object):
    def __init__(self):
        pass

    def __call__(self, geom):
        xs = np.random.randint(1, 20) * 0.1
        ys = np.random.randint(1, 20) * 0.1
        return scale(geom, xfact=xs, yfact=ys, origin='centroid')


class Augment(object):
    def __init__(self):
        self.r = Rotate()
        self.d = Deform()
        self.s = Scale()

    def __call__(self, geom):
        r_geom = self.r(geom)
        d_geom = self.d(geom)
        s_geom = self.s(geom)

        return (geom, r_geom, d_geom, s_geom), ('o', 'r', 'd', 's')


class DatasetGenerator(object):
    def __init__(self, ttfs: list, letters: list):
        self.ttfs = ttfs
        self.letters = letters

    def __call__(self, transform: object = None):
        df = pd.DataFrame()
        geoms, types, values = [], [], []
        for ttf in self.ttfs:
            face = freetype.Face(ttf)
            face.set_char_size(32 * 64)
            for k, v in self.letters.items():
                face.load_char(k, freetype.FT_LOAD_DEFAULT
                               | freetype.FT_LOAD_NO_BITMAP)
                slot = face.glyph
                points = slot.outline.points
                contours = slot.outline.contours
                # Construct polygons with holes
                rings = []
                start_idx = 0
                for idx, end_idx in enumerate(contours):
                    if idx == len(contours) - 1:
                        contour = points[start_idx:]
                    else:
                        contour = points[start_idx:end_idx + 1]
                        start_idx = end_idx + 1

                    if len(contour) >= 3:
                        rings.append(contour)

                geom = Polygon(rings[0],
                               [ring for ring in rings[1:]])
                # Sanity check
                eps = 0.001
                if not geom.is_valid:
                    geom = geom.buffer(eps)

                if geom.is_empty or not isinstance(geom, Polygon):
                    continue
                # Affine trans
                if transform:
                    t_geom, t_type = transform(geom)
                else:
                    t_geom = geom
                    t_type = 'o'

                if isinstance(t_geom, Iterable):
                    t_geom = list(t_geom)
                    t_type = list(t_type)
                else:
                    t_geom = [t_geom]
                    t_type = [t_type]

                geoms += t_geom
                types += t_type
                values += [str(v)] * len(t_geom)

        df["geom"] = geoms
        df["type"] = types
        df["value"] = values
        return df


# Glyph Geom
path = '../../dataset/'
ttfs = glob.glob(path + "ttfs/*")
file_path = 'data/glyph/'

serif_ttfs = glob.glob(ttfs[0] + '/*')
serif_ttfs, test_serif_ttfs = train_test_split(serif_ttfs,
                                               test_size=0.2)

sans_serif_ttfs = glob.glob(ttfs[1] + '/*')
sans_serif_ttfs, \
    test_sans_serif_ttfs = train_test_split(sans_serif_ttfs,
                                            test_size=0.2)

# Dataset Transforms
digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
          '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

sans_serif_set = DatasetGenerator(sans_serif_ttfs, digits)
sans_df = sans_serif_set(transform=Augment())
sans_df.to_pickle(file_path + "sans_trans.pkl")

test_sans_serif_set = DatasetGenerator(test_sans_serif_ttfs, digits)
test_sans_df = test_sans_serif_set(transform=Augment())
test_sans_df.to_pickle(file_path + "test_sans_trans.pkl")

serif_set = DatasetGenerator(serif_ttfs, digits)
serif_df = serif_set(transform=Augment())
serif_df.to_pickle(file_path + "serif_trans.pkl")

test_serif_set = DatasetGenerator(test_serif_ttfs, digits)
test_serif_df = test_serif_set(transform=Augment())
test_serif_df.to_pickle(file_path + "test_serif_trans.pkl")

# Dataset Similar
similar_glyphs = {'0': 0, 'O': 1,
                  '1': 2, 'I': 3,
                  '2': 4, 'Z': 5,
                  '8': 6, 'B': 7}

sans_serif_set = DatasetGenerator(sans_serif_ttfs, similar_glyphs)
sans_df = sans_serif_set(transform=Augment())
sans_df.to_pickle(file_path + "sans_similar.pkl")

test_sans_serif_set = DatasetGenerator(test_sans_serif_ttfs, similar_glyphs)
test_sans_df = test_sans_serif_set(transform=Augment())
test_sans_df.to_pickle(file_path + "test_sans_similar.pkl")

serif_set = DatasetGenerator(serif_ttfs, similar_glyphs)
serif_df = serif_set(transform=Augment())
serif_df.to_pickle(file_path + "serif_similar.pkl")

test_serif_set = DatasetGenerator(test_serif_ttfs, similar_glyphs)
test_serif_df = test_serif_set(transform=Augment())
test_serif_df.to_pickle(file_path + "test_serif_similar.pkl")
