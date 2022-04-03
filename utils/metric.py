import torch
import numpy as np

class Metrics(object):
    def __init__(self, num_classes, category=None):
        self.num_classes = num_classes
        self.cm = torch.zeros(self.num_classes,
                              self.num_classes,
                              dtype=torch.int64)
        self.cum_loss = []
        self.category = category
        
    def update_cm(self, pred, target):
        pred = torch.argmax(pred, 1)
        idx = torch.stack((target, pred)).T
        
        for i in idx:
            self.cm[i[0], i[1]] += 1
            
    def update_loss(self, loss):
        self.cum_loss.append(loss.item())  
        
    def avg_loss(self):
        return torch.mean(torch.tensor(self.cum_loss))       

    def accuracy(self):
        return torch.sum(torch.diagonal(self.cm, 0)) / torch.sum(self.cm)
    
