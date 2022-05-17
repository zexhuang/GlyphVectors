import torch
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
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=32,
                      kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True), 
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(in_channels=32, 
                      out_channels=64,
                      kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, 
                      out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, 
                      out_features=out_channels),
        )
        
    def forward(self, x):
        x = self.cnn_block(x)
        x = torch.mean(x, -1)  # Global Avg Pooling
        return self.fc(x)
    

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
            nn.Linear(32, out_channels)
        )
        
    def forward(self, x):
        return self.rho(x)
    
class Deepset(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 embedding):
        super().__init__()
        self.equi_layer1 = Phi(in_channels, 32)         # Equivariant Layer
        self.equi_layer2 = Phi(32, embedding)           # Equivariant Layer
        self.inva_layer = Rho(embedding, out_channels)  # Set-invariant Layer
        
    def forward(self, x):
        x = self.equi_layer1(x)  
        x = self.equi_layer2(x)
        x, _ = torch.max(x, -1, keepdim=True)           # Global Max Pooling
        x = x.squeeze()
        return self.inva_layer(x)  