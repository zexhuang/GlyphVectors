import numpy as np
import torch
from torch.utils.data import Dataset

class PolygonDataset(Dataset):
    def __init__(self, root_dir, dataset, fixed_tensor=False, transform=None):
        super(Dataset).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.sample_dict = np.load(self.root_dir, allow_pickle=True)
        
        if fixed_tensor:
            self.geoms = self.sample_dict['fixed_size_geoms']
        else:
            self.geoms = self.sample_dict['geoms']
        
        if dataset == 'archae':
            self.feature_type = self.sample_dict['feature_type']
        else:
            self.feature_type = self.sample_dict['building_type']
            
    def __len__(self):
        return len(self.feature_type)
    
    def __getitem__(self, idx):
        geoms = torch.from_numpy(self.geoms[idx]).T
        feature_type = torch.tensor(self.feature_type[idx])
        
        if self.transform:
            pass
        
        sample = {'geoms': geoms,
                  'feature_type': feature_type}
        return sample
    