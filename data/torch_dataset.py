import torch
from torch.utils.data import Dataset

import numpy as np

class LetterVectors(Dataset):
    def __init__(self, root_dir, transform=None):
        super(Dataset).__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.geom = np.load(self.root_dir, allow_pickle=True)["geom"]
        self.value = np.load(self.root_dir, allow_pickle=True)["value"]
        
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
        
        value = torch.tensor(int(self.value[idx]))
        
        if self.transform:
            geom = self.transform(geom)
        
        sample = {'geom': geom,
                  'value': value}
        return sample
    