import numpy as np
import torch
from torch.utils.data import Dataset

class LetterVectors(Dataset):
    def __init__(self, root_dir, transform=None):
        super(Dataset).__init__()
        self.root_dir = root_dir
        self.transform = transform
        
        self.data = np.load(self.root_dir, allow_pickle=True)
        self.samples = self.data['data']
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):    
        geom = torch.from_numpy(self.samples[idx]['geom'])
        char = torch.tensor(int(self.samples[idx]['char']))
        
        if self.transform:
            geom = self.transform(geom)
        
        sample = {'geom': geom.T,
                  'char': char}
        return sample
    