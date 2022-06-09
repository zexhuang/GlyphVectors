from tkinter.messagebox import NO
from matplotlib.pyplot import axis
import torch
from torch.utils.data import Dataset

import numpy as np

class GlyphGeom(Dataset):
    def __init__(self, data_dir, 
                 dataframe=None, 
                 transform=None):
        self.data_dir = data_dir
        self.transform = transform

        if dataframe is None:
            dataframe = np.load(self.data_dir, allow_pickle=True)

        self.geom = dataframe["geom"].to_numpy()
        self.value = dataframe["value"].to_numpy()
        self.type = dataframe["type"].to_numpy()
            
    def __len__(self):
        return len(self.value)
    
    def __getitem__(self, idx):    
        geom = self.geom[idx]
        
        exter = np.asarray(geom.exterior.coords.xy)
        exter = torch.from_numpy(exter)
        exter = torch.cat((exter, torch.zeros(1, exter.size(-1))), dim=0)
        
        inters = list(geom.interiors)
        inter = [np.asarray(i.coords) for i in inters]
        if inter:
            inters = torch.from_numpy(np.concatenate(inter, axis=0)).T
            inters = torch.cat((inters, torch.ones(1, inters.size(-1))), dim=0)
            geom = torch.cat((exter, inters), dim=-1)
        else:
            geom = exter
        
        if self.transform:
            geom = self.transform(geom)
            
        value = torch.tensor(int(self.value[idx]))    
        
        sample = {'geom': geom,
                  'value': value}
        return sample

    
    