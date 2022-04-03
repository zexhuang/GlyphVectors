import torch
from torch import nn 

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