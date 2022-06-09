import torch
class ToFixedTensor(object):
    def __init__(self, embed_size):
        self.embed_size = embed_size
        
    def __call__(self, sample):
        n_dim = sample.size(0)      # feature dim
        m_dim = sample.size(1)      # number dim
        fixed_tensor = torch.zeros(n_dim, self.embed_size)
        if m_dim > self.embed_size:
            sample = sample[:, :self.embed_size]
        else:
            fixed_tensor[:, :m_dim] = sample
        return fixed_tensor