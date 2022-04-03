import numpy as np
import torch
from torch.utils.data import Dataset

class ArchaeDataset(Dataset):
    def __init__(self, root_dir, fixed_tensor=False, transform=None):
        super(Dataset).__init__()
        self.root_dir = root_dir
        self.fixed_tensor = fixed_tensor
        self.transform = transform
        self.sample_dict = np.load(self.root_dir, allow_pickle=True)

    def __len__(self):
        return len(self.sample_dict['feature_type'])
    
    def __getitem__(self, idx):
        if self.fixed_tensor:
            geoms = self.sample_dict['fixed_size_geoms'][idx]
        else:
            geoms = self.sample_dict['geoms'][idx]
        
        feature_type = self.sample_dict['feature_type'][idx] 
        
        if self.transform:
            pass
        
        sample = {'geoms': torch.tensor(geoms).float().transpose(0, -1), 
                  'feature_type': torch.tensor(feature_type)}
        
        return sample
    