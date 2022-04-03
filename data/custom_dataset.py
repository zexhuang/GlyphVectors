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
        
        if self.fixed_tensor:
            self.geoms = torch.tensor(self.sample_dict['fixed_size_geoms'])
        else:
            self.geoms = torch.tensor(self.sample_dict['geoms'])
             
        self.feature_type = torch.tensor(self.sample_dict['feature_type'])
            
    def __len__(self):
        return len(self.sample_dict['feature_type'])
    
    def __getitem__(self, idx):
        geoms = self.geoms[idx].float().transpose(0, -1)
        feature_type = self.feature_type[idx] 
        
        if self.transform:
            pass
        
        sample = {'geoms': geoms,
                  'feature_type': feature_type}
        return sample
    