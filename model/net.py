import torch
from torch import nn 
from torch.nn import functional as F

from config.config_loader import load_config
config = load_config()

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=32,
                      kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(in_channels=32, 
                      out_channels=64,
                      kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=out_channels),
        )
        
    def forward(self, x):
        x = self.cnn_block(x)
        x = torch.mean(x, -1) # global avg pooling
        return self.fc(x)
    

# def nbr_feature(x):
#     x_j = torch.roll(x, 1, dims=-1)
#     x_k = torch.roll(x, -1, dims=-1)
    
#     edge_j = torch.cat((x, x_j - x), dim=1)
#     edge_k = torch.cat((x, x_k - x), dim=1)
#     return torch.stack((edge_j, edge_k), dim=-1)
    

# class EdgeGCN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels*2, 64, kernel_size=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv2d(64, 64, kernel_size=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(negative_slope=0.2)         
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=64, out_features=32),
#             nn.BatchNorm1d(32),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Linear(in_features=32, out_features=out_channels),
#         )
        
#     def forward(self, x):
#         x = nbr_feature(x) 
#         x = self.conv(x)
#         x = x.max(dim=-1, keepdim=False)[0] 
        
#         x = torch.mean(x, dim=-1, keepdim=False)
#         return self.classifier(x)