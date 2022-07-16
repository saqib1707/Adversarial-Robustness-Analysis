import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as TVtransforms
import torchvision.datasets as TVdatasets


class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layer1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)    
        )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)    
        )
        
    def forward(self, x):
        if x.shape[1] == 1:
            out = self.conv_layer1(x)
            out = out.view(-1, 64 * 4 * 4)
            out = self.fc_layer1(out)
        elif x.shape[1] == 3:
            out = self.conv_layer2(x)
            out = out.view(-1, 64 * 5 * 5)
            out = self.fc_layer2(out)
        
        return out