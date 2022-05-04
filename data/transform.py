import torch

class ToCoordinate(object):    
    def __call__(self, sample):
        coordinate = sample[0:2, :-1]
        return coordinate
    
class ToFixedTensor(object):
    def __init__(self, embed_size):
        self.embed_size = embed_size
        
    def __call__(self, sample):
        fixed_tensor = torch.zeros(self.embed_size, 2)
        dim = sample.size(0)
        fixed_tensor[:dim, :2] = sample
        return fixed_tensor