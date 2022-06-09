import torch
import math
from torch import nn 
from torch.nn import functional as F

from config.config_loader import load_config
config = load_config()


"""CNN"""
class CNN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=32,
                      kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), 
            nn.MaxPool1d(3, stride=3))
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, 
                      out_channels=64,
                      kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=3))
            
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, 
                      out_channels=128,
                      kernel_size=5),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
            
        self.cls = nn.Sequential(
            nn.Linear(in_features=128, 
                      out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, 
                      out_features=out_channels))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        return self.cls(x)
    

"""Deepsets"""
class Phi(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 
                               out_channels, 
                               kernel_size=1, 
                               bias=False)
        self.conv2 = nn.Conv1d(in_channels, 
                               out_channels, 
                               kernel_size=1, 
                               bias=False)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        xm, _ = x.max(-1, keepdim=True)
        xm = self.bn1(self.conv1(xm))
        
        x = self.bn2(self.conv2(x))
        x = F.relu(x - xm, inplace=True)
        return x

class Rho(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(in_channels, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_channels))
        
    def forward(self, x):
        return self.rho(x)
    
class Deepset(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 embedding):
        super(Deepset, self).__init__()
        self.equi_layer1 = Phi(in_channels, 32)         # Equivariant Layer
        self.equi_layer2 = Phi(32, 64)                  # Equivariant Layer
        self.equi_layer3 = Phi(64, embedding)           # Equivariant Layer
        self.inva_layer = Rho(embedding, out_channels)  # Set-invariant Layer
        
    def forward(self, x):
        x = self.equi_layer1(x)  
        x = self.equi_layer2(x)
        x = self.equi_layer3(x)
        
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        return self.inva_layer(x)  


"""GCNN"""
def edge_feature(x):
    x1 = torch.roll(x, shifts=-1, dims=-1)
    feat1 = torch.cat([x, x-x1], dim=1)
    
    x2 = torch.roll(x, shifts=1, dims=-1)
    feat2 = torch.cat([x, x-x2], dim=1)
    
    x = torch.stack([feat1, feat2], dim=-1)
    return x
    
class GCNN(nn.Module):
    def __init__(self, in_channels, out_channels, embedding):
        super(GCNN, self).__init__()
        self.embedding = embedding
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels*2, 
                                             32, 
                                             kernel_size=1, 
                                             bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32*2, 
                                             64, 
                                             kernel_size=1, 
                                             bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(32 + 64, 
                                             embedding, 
                                             kernel_size=1, 
                                             bias=False),
                                   nn.BatchNorm1d(embedding),
                                   nn.ReLU(inplace=True))
        self.cls = nn.Sequential(
            nn.Linear(embedding, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_channels))
        
    def forward(self, x):
        x = edge_feature(x)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = edge_feature(x1)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]   
        
        x = torch.cat((x1, x2), dim=1) 
        x = self.conv3(x)
        
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        return self.cls(x)
    
    
"""Set Transformer"""
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.BatchNorm1d(dim_V)
            self.ln1 = nn.BatchNorm1d(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O.permute(0, -1, 1)).permute(0, -1, 1)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O.permute(0, -1, 1)).permute(0, -1, 1)
        return O
    
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)
    
class SetTransformer(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden, 
                 num_heads=4, num_inds=32, ln=True):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                                 ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        
        self.cls = nn.Sequential(
            nn.Linear(dim_hidden, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, dim_output))
        
    def forward(self, x):
        x = torch.permute(x, (0, -1, 1))
        x = self.enc(x)

        x = torch.permute(x, (0, -1, 1))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = self.cls(x)
        return x